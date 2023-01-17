/*
 * Logger.cpp
 *
 *  Created on: 17 Jan 2023
 *      Author: luk
 */

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "Logger.h"

Logger::Logger()
 : lastWritten(-1),
   writeThread(0)
{
    std::string deviceId = "#1";

    depth_compress_buf_size = 640 * 480 * sizeof(int16_t) * 4;
    depth_compress_buf = (uint8_t*)malloc(depth_compress_buf_size);

    encodedImage = 0;

    writing.assignValue(false);

    latestDepthIndex.assignValue(-1);
    latestImageIndex.assignValue(-1);

    for(int i = 0; i < 10; i++)
    {
        uint8_t * newImage = (uint8_t *)calloc(640 * 480 * 3, sizeof(uint8_t));
        imageBuffers[i] = std::pair<uint8_t *, int64_t>(newImage, 0);
    }

    for(int i = 0; i < 10; i++)
    {
        uint8_t * newDepth = (uint8_t *)calloc(640 * 480 * 2, sizeof(uint8_t));
        uint8_t * newImage = (uint8_t *)calloc(640 * 480 * 3, sizeof(uint8_t));
        frameBuffers[i] = std::pair<std::pair<uint8_t *, uint8_t *>, int64_t>(std::pair<uint8_t *, uint8_t *>(newDepth, newImage), 0);
    }

    setupDevice(deviceId);
}

Logger::~Logger()
{
    if(m_device)
    {
        m_device->stopDepthStream();
        m_device->stopImageStream();
    }

    free(depth_compress_buf);

    writing.assignValue(false);

    dataThread->join();
    writeThread->join();

    if(encodedImage != 0)
    {
        cvReleaseMat(&encodedImage);
    }

    for(int i = 0; i < 10; i++)
    {
        free(imageBuffers[i].first);
    }

    for(int i = 0; i < 10; i++)
    {
        free(frameBuffers[i].first.first);
        free(frameBuffers[i].first.second);
    }
}


void Logger::setupDevice(const std::string & deviceId)
{
    dataThread = new boost::thread(boost::bind(&Logger::readFrames,
                                                this));
}


void Logger::readFrames()
{
    boost::filesystem::path the_path( "data" );

    int count = std::count_if(
            boost::filesystem::directory_iterator(the_path),
            boost::filesystem::directory_iterator(),
            static_cast<bool(*)(const boost::filesystem::path&)>(boost::filesystem::is_regular_file) );

    std::cout << count << std::endl;

    boost::this_thread::sleep_for(boost::chrono::milliseconds(2000));

    for (int i = 1; i < count; ++i)
    {
        char str[5];
        snprintf (str, 5, "%04d", i);
        std::cout << std::string(str) << std::endl;
        cv::Mat image = cv::imread("data/scan_" + std::string(str) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(640, 480), cv::INTER_LINEAR);

        cv::Mat texture;
        cv::Mat depth;
        cv::Mat other;
        cv::extractChannel(resized, texture, 0);
        cv::extractChannel(resized, depth, 2);
        cv::extractChannel(resized, other, 1);

        cv::Mat triple;
        std::vector<cv::Mat> channels;
        channels.push_back(texture);
        channels.push_back(texture);
        channels.push_back(texture);
        cv::merge(channels, triple);

        //other = 256 - other;
        //depth = 256 - depth;
        //other.setTo(0, other == 255);
        //depth.setTo(0, depth == 255);
        cv::Mat ddepth;
        std::vector<cv::Mat> cchannels;
        cchannels.push_back(other);
        cchannels.push_back(depth);
        cv::merge(cchannels, ddepth);

        //std::cout << triple.channels() << std::endl;
        //std::cout << triple.cols << std::endl;
        //std::cout << triple.rows << std::endl;

        boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration duration(time.time_of_day());
        m_lastImageTime = duration.total_microseconds();
        m_lastDepthTime = m_lastImageTime;

        int textureBufferIndex = (latestImageIndex.getValue() + 1) % 10;

        //imageBuffers[textureBufferIndex].first = triple.data;
        memcpy(imageBuffers[textureBufferIndex].first, triple.data, 640 * 480 * 3);
        imageBuffers[textureBufferIndex].second = m_lastImageTime;

        latestImageIndex++;

        int depthBufferIndex = (latestDepthIndex.getValue() + 1) % 10;

        //frameBuffers[depthBufferIndex].first.first = depth.data;
        memcpy(frameBuffers[depthBufferIndex].first.first, ddepth.data, 640 * 480 * 2);
        frameBuffers[depthBufferIndex].second = m_lastDepthTime;

        int lastImageVal = latestImageIndex.getValue();
        lastImageVal %= 10;

        memcpy(frameBuffers[depthBufferIndex].first.second, imageBuffers[lastImageVal].first, 640 * 480 * 3);

        latestDepthIndex++;

        boost::this_thread::sleep_for(boost::chrono::milliseconds(20));
    }
}

void Logger::startSynchronization()
{
    if(m_device->isSynchronizationSupported() &&
       !m_device->isSynchronized() &&
       m_device->getImageOutputMode().nFPS == m_device->getDepthOutputMode().nFPS &&
       m_device->isImageStreamRunning() &&
       m_device->isDepthStreamRunning())
    {
        m_device->setSynchronization(true);
    }
}

void Logger::stopSynchronization()
{
    if(m_device->isSynchronizationSupported() && m_device->isSynchronized())
    {
        m_device->setSynchronization(false);
    }
}

void Logger::encodeJpeg(cv::Vec<unsigned char, 3> * rgb_data)
{
    cv::Mat3b rgb(480, 640, rgb_data, 1920);

    IplImage * img = new IplImage(rgb);

    int jpeg_params[] = {CV_IMWRITE_JPEG_QUALITY, 90, 0};

    if(encodedImage != 0)
    {
        cvReleaseMat(&encodedImage);
    }

    encodedImage = cvEncodeImage(".jpg", img, jpeg_params);

    delete img;
}

void Logger::imageCallback(boost::shared_ptr<openni_wrapper::Image> image, void * cookie)
{
	boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration(time.time_of_day());
	m_lastImageTime = duration.total_microseconds();

    int bufferIndex = (latestImageIndex.getValue() + 1) % 10;

    image->fillRGB(image->getWidth(), image->getHeight(), reinterpret_cast<unsigned char*>(imageBuffers[bufferIndex].first), 640 * 3);

    imageBuffers[bufferIndex].second = m_lastImageTime;

    latestImageIndex++;
}

void Logger::depthCallback(boost::shared_ptr<openni_wrapper::DepthImage> depth_image, void * cookie)
{
	boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration(time.time_of_day());
	m_lastDepthTime = duration.total_microseconds();
    
	int bufferIndex = (latestDepthIndex.getValue() + 1) % 10;

    depth_image->fillDepthImageRaw(depth_image->getWidth(), depth_image->getHeight(), reinterpret_cast<unsigned short *>(frameBuffers[bufferIndex].first.first), 640 * 2);

    frameBuffers[bufferIndex].second = m_lastDepthTime;

    int lastImageVal = latestImageIndex.getValue();

    if(lastImageVal == -1)
    {
        return;
    }

    lastImageVal %= 10;

    memcpy(frameBuffers[bufferIndex].first.second, imageBuffers[lastImageVal].first, 640 * 480 * 3);

    latestDepthIndex++;
}

void Logger::startWriting(std::string filename)
{
    assert(!writeThread && !writing.getValue());

    this->filename = filename;

    writing.assignValue(true);

    writeThread = new boost::thread(boost::bind(&Logger::writeData,
                                               this));
}

void Logger::stopWriting()
{
    assert(writeThread && writing.getValue());

    writing.assignValue(false);

    writeThread->join();

    writeThread = 0;
}

void Logger::writeData()
{
    /**
     * int32_t at file beginning for frame count
     */
    FILE * logFile = fopen(filename.c_str(), "wb+");

    int32_t numFrames = 0;

    fwrite(&numFrames, sizeof(int32_t), 1, logFile);

    while(writing.getValueWait(1))
    {
        int lastDepth = latestDepthIndex.getValue();

        if(lastDepth == -1)
        {
            continue;
        }

        int bufferIndex = lastDepth % 10;

        if(bufferIndex == lastWritten)
        {
            continue;
        }

        unsigned long compressed_size = depth_compress_buf_size;
        boost::thread_group threads;

        threads.add_thread(new boost::thread(compress2,
                                             depth_compress_buf,
                                             &compressed_size,
                                             (const Bytef*)frameBuffers[bufferIndex].first.first,
                                             640 * 480 * sizeof(short),
                                             Z_BEST_SPEED));

        threads.add_thread(new boost::thread(boost::bind(&Logger::encodeJpeg,
                                                         this,
                                                         (cv::Vec<unsigned char, 3> *)frameBuffers[bufferIndex].first.second)));

        threads.join_all();

        int32_t depthSize = compressed_size;
        int32_t imageSize = encodedImage->width;

        /**
         * Format is:
         * int64_t: timestamp
         * int32_t: depthSize
         * int32_t: imageSize
         * depthSize * unsigned char: depth_compress_buf
         * imageSize * unsigned char: encodedImage->data.ptr
         */

        fwrite(&frameBuffers[bufferIndex].second, sizeof(int64_t), 1, logFile);
        fwrite(&depthSize, sizeof(int32_t), 1, logFile);
        fwrite(&imageSize, sizeof(int32_t), 1, logFile);
        fwrite(depth_compress_buf, depthSize, 1, logFile);
        fwrite(encodedImage->data.ptr, imageSize, 1, logFile);

        numFrames++;

        lastWritten = bufferIndex;
    }

    fseek(logFile, 0, SEEK_SET);
    fwrite(&numFrames, sizeof(int32_t), 1, logFile);

    fflush(logFile);
    fclose(logFile);
}


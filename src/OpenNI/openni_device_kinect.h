/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011 Willow Garage, Inc.
 *    Suat Gedikli <gedikli@willowgarage.com>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENNI_DEVICE_KINECT__
#define __OPENNI_DEVICE_KINECT__

#include "openni_device.h"
#include "openni_driver.h"
#include "openni_image_bayer_grbg.h"

namespace openni_wrapper
{

/**
 * @brief Concrete implementation of the interface OpenNIDevice for a MS Kinect device.
 * @author Suat Gedikli
 * @date 02.january 2011
 */
class DeviceKinect : public OpenNIDevice
{
  friend class OpenNIDriver;
public:
  DeviceKinect (xn::Context& context, const xn::NodeInfo& device_node, const xn::NodeInfo& image_node, const xn::NodeInfo& depth_node, const xn::NodeInfo& ir_node) throw (OpenNIException);
  virtual ~DeviceKinect () throw ();

  inline void setDebayeringMethod (const ImageBayerGRBG::DebayeringMethod& debayering_method) throw ();
  inline const ImageBayerGRBG::DebayeringMethod& getDebayeringMethod () const throw ();
  
  // these capabilities are not supported for kinect
  virtual void setSynchronization (bool on_off) throw (OpenNIException);
  virtual bool isSynchronized () const throw (OpenNIException);
  virtual bool isSynchronizationSupported () const throw ();

  virtual bool isDepthCropped () const throw (OpenNIException);
  virtual void setDepthCropping (unsigned x, unsigned y, unsigned width, unsigned height) throw (OpenNIException);
  virtual bool isDepthCroppingSupported () const throw ();

protected:
  virtual boost::shared_ptr<Image> getCurrentImage (boost::shared_ptr<xn::ImageMetaData> image_meta_data) const throw ();
  virtual void enumAvailableModes () throw (OpenNIException);
  virtual bool isImageResizeSupported (unsigned input_width, unsigned input_height, unsigned output_width, unsigned output_height) const throw ();
  ImageBayerGRBG::DebayeringMethod debayering_method_;
};

void DeviceKinect::setDebayeringMethod (const ImageBayerGRBG::DebayeringMethod& debayering_method) throw ()
{
  debayering_method_ = debayering_method;
}

const ImageBayerGRBG::DebayeringMethod& DeviceKinect::getDebayeringMethod () const throw ()
{
  return debayering_method_;
}
} // namespace

#endif // __OPENNI_DEVICE_KINECT__

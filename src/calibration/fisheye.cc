/*M/////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// Self Header
#include <calibration/fisheye.h>

// System
#include <iostream>

// Glog
#include <glog/logging.h>

// OpenCV2
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

// using namespace cv;
using namespace cv;
using namespace std;

struct IntrinsicParams {
  Vec2d f;
  Vec2d c;
  Vec4d k;
  double alpha;
  std::vector<uchar> isEstimate;

  IntrinsicParams();
  IntrinsicParams(Vec2d f, Vec2d c, Vec4d k, double alpha = 0);
  IntrinsicParams operator+(const Mat& a);
  IntrinsicParams& operator=(const Mat& a);
  void Init(const cv::Vec2d& f, const cv::Vec2d& c,
            const cv::Vec4d& k = Vec4d(0, 0, 0, 0), const double& alpha = 0);
};

struct JacobianRow {
  Vec2d df, dc;
  Vec4d dk;
  Vec3d dom, dT;
  double dalpha;
};

IntrinsicParams::IntrinsicParams()
    : f(Vec2d::all(0)),
      c(Vec2d::all(0)),
      k(Vec4d::all(0)),
      alpha(0),
      isEstimate(9, 0) {}

IntrinsicParams::IntrinsicParams(Vec2d _f, Vec2d _c, Vec4d _k, double _alpha)
    : f(_f), c(_c), k(_k), alpha(_alpha), isEstimate(9, 0) {}

IntrinsicParams IntrinsicParams::operator+(const Mat& a) {
  CV_Assert(a.type() == CV_64FC1);
  IntrinsicParams tmp;
  const double* ptr = a.ptr<double>();

  int j = 0;
  tmp.f[0] = this->f[0] + (isEstimate[0] ? ptr[j++] : 0);
  tmp.f[1] = this->f[1] + (isEstimate[1] ? ptr[j++] : 0);
  tmp.c[0] = this->c[0] + (isEstimate[2] ? ptr[j++] : 0);
  tmp.alpha = this->alpha + (isEstimate[4] ? ptr[j++] : 0);
  tmp.c[1] = this->c[1] + (isEstimate[3] ? ptr[j++] : 0);
  tmp.k[0] = this->k[0] + (isEstimate[5] ? ptr[j++] : 0);
  tmp.k[1] = this->k[1] + (isEstimate[6] ? ptr[j++] : 0);
  tmp.k[2] = this->k[2] + (isEstimate[7] ? ptr[j++] : 0);
  tmp.k[3] = this->k[3] + (isEstimate[8] ? ptr[j++] : 0);

  tmp.isEstimate = isEstimate;
  return tmp;
}

IntrinsicParams& IntrinsicParams::operator=(const Mat& a) {
  CV_Assert(a.type() == CV_64FC1);
  const double* ptr = a.ptr<double>();

  int j = 0;

  this->f[0] = isEstimate[0] ? ptr[j++] : 0;
  this->f[1] = isEstimate[1] ? ptr[j++] : 0;
  this->c[0] = isEstimate[2] ? ptr[j++] : 0;
  this->c[1] = isEstimate[3] ? ptr[j++] : 0;
  this->alpha = isEstimate[4] ? ptr[j++] : 0;
  this->k[0] = isEstimate[5] ? ptr[j++] : 0;
  this->k[1] = isEstimate[6] ? ptr[j++] : 0;
  this->k[2] = isEstimate[7] ? ptr[j++] : 0;
  this->k[3] = isEstimate[8] ? ptr[j++] : 0;

  return *this;
}

void IntrinsicParams::Init(const cv::Vec2d& _f, const cv::Vec2d& _c,
                           const cv::Vec4d& _k, const double& _alpha) {
  this->c = _c;
  this->f = _f;
  this->k = _k;
  this->alpha = _alpha;
}

cv::Mat NormalizePixels(const Mat& imagePoints, const IntrinsicParams& param) {
  CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

  Mat distorted((int)imagePoints.total(), 1, CV_64FC2), undistorted;
  const Vec2d* ptr = imagePoints.ptr<Vec2d>();
  Vec2d* ptr_d = distorted.ptr<Vec2d>();
  for (size_t i = 0; i < imagePoints.total(); ++i) {
    ptr_d[i] =
        (ptr[i] - param.c).mul(Vec2d(1.0 / param.f[0], 1.0 / param.f[1]));
    ptr_d[i][0] -= param.alpha * ptr_d[i][1];
  }
  cv::fisheye::undistortPoints(distorted, undistorted, Matx33d::eye(), param.k);
  return undistorted;
}

void CreateHomogeneousPoints(const Mat& m, Mat& m_homo) {
  int point_num = m.cols;
  if (m.rows < 3) {
    vconcat(m, Mat::ones(1, point_num, CV_64FC1), m_homo);
  } else {
    m_homo = m.clone();
  }
  divide(m_homo, Mat::ones(3, 1, CV_64FC1) * m_homo.row(2), m_homo);
}

void NormalizePoints(const Mat& pnt_homo, Mat& pnt_normalized,
                     Mat& inv_norm_mat) {
  {
    Mat mx_row = pnt_homo.row(0).clone();
    Mat my_row = pnt_homo.row(1).clone();
    double mx_mean = mean(mx_row)[0];
    double my_mean = mean(my_row)[0];
    Mat normalized_mx_row = mx_row - mx_mean;
    Mat normalized_my_row = my_row - my_mean;
    double x_scale = mean(abs(normalized_mx_row))[0];
    double y_scale = mean(abs(normalized_my_row))[0];

    Mat norm_mat(Matx33d(1 / x_scale, 0.0, -mx_mean / x_scale, 0.0, 1 / y_scale,
                         -my_mean / y_scale, 0.0, 0.0, 1.0));
    inv_norm_mat =
        Mat(Matx33d(x_scale, 0, mx_mean, 0, y_scale, my_mean, 0, 0, 1));
    pnt_normalized = norm_mat * pnt_homo;
  }
}

Mat ComputeHomographyViaDLT(const Mat& pnt_to_be_projected,
                            const Mat& pnt_destnation) {
  // Build linear system L.
  int point_num = pnt_to_be_projected.cols;
  Mat L = Mat::zeros(2 * point_num, 9, CV_64FC1);
  for (int i = 0; i < point_num; ++i) {
    for (int j = 0; j < 3; j++) {
      L.at<double>(2 * i, j) = pnt_destnation.at<double>(j, i);
      L.at<double>(2 * i, j + 6) = -pnt_to_be_projected.at<double>(0, i) *
                                   pnt_destnation.at<double>(j, i);
      L.at<double>(2 * i + 1, j + 3) = pnt_destnation.at<double>(j, i);
      L.at<double>(2 * i + 1, j + 6) = -pnt_to_be_projected.at<double>(1, i) *
                                       pnt_destnation.at<double>(j, i);
    }
  }

  // Solve H via SVD.
  Mat H;
  {
    if (point_num > 4) {
      L = L.t() * L;
    }
    SVD svd(L);
    Mat h_vector = svd.vt.row(8) / svd.vt.row(8).at<double>(8);
    H = h_vector.reshape(1, 3);
    // H = inv_norm_mat * Hrem;
  }
  return H;
}

void RefineHomographyViaGaussNewton(const Mat& pnt_to_be_projected,
                                    const Mat& pnt_destnation, Mat& H) {
  int point_num = pnt_to_be_projected.cols;
  if (point_num > 4) {
    Mat h_vector = H.reshape(1, 9)(Rect(0, 0, 1, 8)).clone();
    for (int iter = 0; iter < 10; iter++) {
      Mat m_proj = H * pnt_destnation;
      Mat MM_homo, m_reproj_err;

      // Calculate Reprojection Error.
      {
        // Scale Adjustment. Dividing both side of eqn by m_proj.z.
        divide(pnt_destnation,
               Mat::ones(3, 1, CV_64FC1) * m_proj(Rect(0, 2, m_proj.cols, 1)),
               MM_homo);
        divide(m_proj,
               Mat::ones(3, 1, CV_64FC1) * m_proj(Rect(0, 2, m_proj.cols, 1)),
               m_proj);

        m_reproj_err =
            m_proj(Rect(0, 0, m_proj.cols, 2)) -
            pnt_to_be_projected(Rect(0, 0, pnt_to_be_projected.cols, 2));
        m_reproj_err = Mat(m_reproj_err.t())
                           .reshape(1, m_reproj_err.cols * m_reproj_err.rows);
      }

      // Create Jacobian.
      Mat J = Mat::zeros(2 * point_num, 8, CV_64FC1);
      {
        Mat MMM2, MMM3;
        multiply(Mat::ones(3, 1, CV_64FC1) * m_proj(Rect(0, 0, m_proj.cols, 1)),
                 MM_homo, MMM2);
        multiply(Mat::ones(3, 1, CV_64FC1) * m_proj(Rect(0, 1, m_proj.cols, 1)),
                 MM_homo, MMM3);

        for (int i = 0; i < point_num; ++i) {
          for (int j = 0; j < 3; ++j) {
            J.at<double>(2 * i, j) = pnt_destnation.at<double>(j, i);
            J.at<double>(2 * i + 1, j + 3) = pnt_destnation.at<double>(j, i);
          }

          for (int j = 0; j < 2; ++j) {
            J.at<double>(2 * i, j + 6) = -MMM2.at<double>(j, i);
            J.at<double>(2 * i + 1, j + 6) = -MMM3.at<double>(j, i);
          }
        }
      }

      // Update Homography Matrix H.
      {
        Mat dh_vec = -(J.t() * J).inv() * (J.t()) * m_reproj_err;
        h_vector = h_vector + dh_vec;
        Mat tmp;
        vconcat(h_vector, Mat::ones(1, 1, CV_64FC1), tmp);
        H = tmp.reshape(1, 3);
      }
    }
  }
}

cv::Mat ComputeHomography(const Mat& m_, const Mat& M_) {
  // Convert input to homogeneous Mat.
  Mat m_homo, M_homo;
  CreateHomogeneousPoints(m_, m_homo);
  CreateHomogeneousPoints(M_, M_homo);

  // Normalize x and y coordinate to make homogeneous.
  Mat m_normalized, inv_norm_mat;
  NormalizePoints(m_homo, m_normalized, inv_norm_mat);

  // Compuate initial homography matrix.
  Mat H = ComputeHomographyViaDLT(m_normalized, M_homo);
  H = inv_norm_mat * H;

  // Refine homography matrix via optimization.
  RefineHomographyViaGaussNewton(m_homo, M_homo, H);

  return H;
}

void subMatrix(const Mat& src, Mat& dst, const std::vector<uchar>& cols,
               const std::vector<uchar>& rows) {
  CV_Assert(src.channels() == 1);

  int nonzeros_cols = cv::countNonZero(cols);
  Mat tmp(src.rows, nonzeros_cols, CV_64F);

  for (int i = 0, j = 0; i < (int)cols.size(); i++) {
    if (cols[i]) {
      src.col(i).copyTo(tmp.col(j++));
    }
  }

  int nonzeros_rows = cv::countNonZero(rows);
  dst.create(nonzeros_rows, nonzeros_cols, CV_64F);
  for (int i = 0, j = 0; i < (int)rows.size(); i++) {
    if (rows[i]) {
      tmp.row(i).copyTo(dst.row(j++));
    }
  }
}

void ConvertVector2dToMat(const vector<cv::Point3f>& objectPoints,
                          const vector<cv::Point2f>& imagePoints,
                          cv::Mat& object_points_mat,
                          cv::Mat& image_points_mat) {
  {
    cv::Mat(objectPoints).convertTo(object_points_mat, CV_64FC3);
    cv::Mat(imagePoints).convertTo(image_points_mat, CV_64FC2);
    if (object_points_mat.rows == 3) {
      object_points_mat = object_points_mat.t();
    }
    if (image_points_mat.rows == 2) {
      image_points_mat = image_points_mat.t();
    }
  }
}

void TransformToNormalizedFrame(const Mat& object_points_mat,
                                Mat& normalized_object_points_mat,
                                cv::Mat& rep_rot, cv::Mat& rep_trans) {
  // Compute COG and Angle via SVD.
  {
    Mat objectPoints = object_points_mat.reshape(1).t();
    Mat objectPointsMean, covObjectPoints;
    calcCovarMatrix(objectPoints, covObjectPoints, objectPointsMean,
                    COVAR_NORMAL | COVAR_COLS);
    SVD svd(covObjectPoints);
    rep_rot = svd.vt;
    if (norm(rep_rot(Rect(2, 0, 1, 2))) < 1e-6) {
      rep_rot = Mat::eye(3, 3, CV_64FC1);
    }
    if (determinant(rep_rot) < 0) {
      rep_rot = -rep_rot;
    }
    // rep_trans = rep_rot * objectPointsMean;
    rep_trans = objectPointsMean;
    int Np = objectPoints.cols;

    // Subtract translation from origin to normalize points.
    normalized_object_points_mat =
        rep_rot.t() * objectPoints +
        (rep_rot.t() * -rep_trans) * Mat::ones(1, Np, CV_64FC1);
  }
}

void ComputeExtrinsicsBasedOnHomography(const Mat& image_points_mat,
                                        const Mat& object_points_mat,
                                        const IntrinsicParams& param,
                                        Mat& board_rot, Mat& board_trans) {
  CV_Assert(!object_points_mat.empty() && object_points_mat.type() == CV_64FC3);
  CV_Assert(!image_points_mat.empty() && image_points_mat.type() == CV_64FC2);

  // Transform points to the normalized coordinate on calibration board.
  Mat object_points_in_normalized_frame, rep_rot, rep_trans;
  TransformToNormalizedFrame(
      object_points_mat, object_points_in_normalized_frame, rep_rot, rep_trans);

  // Compute Homography.
  Mat H;
  {
    Mat imagePointsNormalized =
        NormalizePixels(image_points_mat, param).reshape(1).t();
    H = ComputeHomography(
        imagePointsNormalized,
        object_points_in_normalized_frame(
            Rect(0, 0, object_points_in_normalized_frame.cols, 2)));
    // Norm of basis vector should be one. (r1, r2)
    double scale = .5 * (norm(H.col(0)) + norm(H.col(1)));
    H = H / scale;
  }

  // Axis and Rotation Matrix generation.
  cv::Mat rep_rot_in_cam;
  {
    cv::Mat u1, u2, u3;
    u1 = H.col(0) / norm(H.col(0));
    u2 = H.col(1).clone() - u1.dot(H.col(1).clone()) * u1;
    u2 = u2 / norm(u2);
    u3 = u1.cross(u2);
    hconcat(u1, u2, rep_rot_in_cam);
    hconcat(rep_rot_in_cam, u3, rep_rot_in_cam);
  }

  // Compute calibration origin from representative origin.
  {
    cv::Mat rep_trans_in_cam = H.col(2).clone();
    board_trans =
        rep_rot_in_cam * rep_rot.t() * (-rep_trans) + rep_trans_in_cam;
    cv::Mat board_rot_mat = rep_rot_in_cam * rep_rot.t();
    Rodrigues(board_rot_mat, board_rot);
  }
}

void FisheyeProjectPoints(InputArray objectPoints, OutputArray imagePoints,
                          InputArray _rvec, InputArray _tvec, Matx33d K,
                          Vec4d D, double alpha, OutputArray jacobian) {
  // will support only 3-channel data now for points
  CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
  imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
  size_t point_num = objectPoints.total();

  CV_Assert(_rvec.total() * _rvec.channels() == 3 &&
            (_rvec.depth() == CV_32F || _rvec.depth() == CV_64F));
  CV_Assert(_tvec.total() * _tvec.channels() == 3 &&
            (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));
  CV_Assert(_tvec.getMat().isContinuous() && _rvec.getMat().isContinuous());

  Vec3d board_rot = _rvec.depth() == CV_32F
                        ? (Vec3d)*_rvec.getMat().ptr<Vec3f>()
                        : *_rvec.getMat().ptr<Vec3d>();
  Vec3d board_trans = _tvec.depth() == CV_32F
                          ? (Vec3d)*_tvec.getMat().ptr<Vec3f>()
                          : *_tvec.getMat().ptr<Vec3d>();

  cv::Vec2d f, c;
  f = Vec2d(K(0, 0), K(1, 1));
  c = Vec2d(K(0, 2), K(1, 2));
  Vec4d k = D;

  JacobianRow* Jn = 0;
  if (jacobian.needed()) {
    int nvars = 2 + 2 + 1 + 4 + 3 + 3;  // f, c, alpha, k, om, T,
    jacobian.create(2 * point_num, nvars, CV_64F);
    Jn = jacobian.getMat().ptr<JacobianRow>(0);
  }

  // Jacobian wrt board rotation.
  Matx<double, 3, 9> dRdom;
  {
    Matx33d board_rot_mat;
    Rodrigues(board_rot, board_rot_mat, dRdom);
  }

  Affine3d aff(board_rot, board_trans);

  const Vec3d* Xd = objectPoints.getMat().ptr<Vec3d>();
  Vec2d* proj_points = imagePoints.getMat().ptr<Vec2d>();

  for (size_t i = 0; i < point_num; ++i) {
    Vec3d Xi = Xd[i];
    Vec3d Y = aff * Xi;

    Vec2d x(Y[0] / Y[2], Y[1] / Y[2]);

    // Angle of the incoming ray:
    double r2 = x.dot(x);
    double r = std::sqrt(r2);
    double theta = atan(r);

    double theta2 = theta * theta;
    double theta3 = theta2 * theta;
    double theta4 = theta2 * theta2;
    double theta5 = theta4 * theta;
    double theta6 = theta3 * theta3;
    double theta7 = theta6 * theta;
    double theta8 = theta4 * theta4;
    double theta9 = theta8 * theta;
    double theta_d =
        theta + k[0] * theta3 + k[1] * theta5 + k[2] * theta7 + k[3] * theta9;
    double inv_r = r > 1e-8 ? 1.0 / r : 1;
    double cdist = r > 1e-8 ? theta_d * inv_r : 1;

    Vec2d xd1 = x * cdist;
    Vec2d xd3(xd1[0] + alpha * xd1[1], xd1[1]);
    Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);
    proj_points[i] = final_point;

    if (jacobian.needed()) {
      double dYdR[] = {Xi[0], Xi[1], Xi[2], 0,     0,     0,     0,    0, 0, 0,
                       0,     0,     Xi[0], Xi[1], Xi[2], 0,     0,    0, 0, 0,
                       0,     0,     0,     0,     Xi[0], Xi[1], Xi[2]};

      Matx33d dYdom_data = Matx<double, 3, 9>(dYdR) * dRdom.t();
      const Vec3d* dYdom = (Vec3d*)dYdom_data.val;

      Matx33d dYdT_data = Matx33d::eye();
      const Vec3d* dYdT = (Vec3d*)dYdT_data.val;

      Vec3d dxdom[2];
      dxdom[0] = (1.0 / Y[2]) * dYdom[0] - x[0] / Y[2] * dYdom[2];
      dxdom[1] = (1.0 / Y[2]) * dYdom[1] - x[1] / Y[2] * dYdom[2];

      Vec3d dxdT[2];
      dxdT[0] = (1.0 / Y[2]) * dYdT[0] - x[0] / Y[2] * dYdT[2];
      dxdT[1] = (1.0 / Y[2]) * dYdT[1] - x[1] / Y[2] * dYdT[2];

      // double r2 = x.dot(x);
      Vec3d dr2dom = 2 * x[0] * dxdom[0] + 2 * x[1] * dxdom[1];
      Vec3d dr2dT = 2 * x[0] * dxdT[0] + 2 * x[1] * dxdT[1];

      // double r = std::sqrt(r2);
      double drdr2 = r > 1e-8 ? 1.0 / (2 * r) : 1;
      Vec3d drdom = drdr2 * dr2dom;
      Vec3d drdT = drdr2 * dr2dT;

      // Angle of the incoming ray:
      // double theta = atan(r);
      double dthetadr = 1.0 / (1 + r2);
      Vec3d dthetadom = dthetadr * drdom;
      Vec3d dthetadT = dthetadr * drdT;

      // double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 +
      // k[3]*theta9;
      double dtheta_ddtheta = 1 + 3 * k[0] * theta2 + 5 * k[1] * theta4 +
                              7 * k[2] * theta6 + 9 * k[3] * theta8;
      Vec3d dtheta_ddom = dtheta_ddtheta * dthetadom;
      Vec3d dtheta_ddT = dtheta_ddtheta * dthetadT;
      Vec4d dtheta_ddk = Vec4d(theta3, theta5, theta7, theta9);

      // double inv_r = r > 1e-8 ? 1.0/r : 1;
      // double cdist = r > 1e-8 ? theta_d / r : 1;
      Vec3d dcdistdom = inv_r * (dtheta_ddom - cdist * drdom);
      Vec3d dcdistdT = inv_r * (dtheta_ddT - cdist * drdT);
      Vec4d dcdistdk = inv_r * dtheta_ddk;

      // Vec2d xd1 = x * cdist;
      Vec4d dxd1dk[2];
      Vec3d dxd1dom[2], dxd1dT[2];
      dxd1dom[0] = x[0] * dcdistdom + cdist * dxdom[0];
      dxd1dom[1] = x[1] * dcdistdom + cdist * dxdom[1];
      dxd1dT[0] = x[0] * dcdistdT + cdist * dxdT[0];
      dxd1dT[1] = x[1] * dcdistdT + cdist * dxdT[1];
      dxd1dk[0] = x[0] * dcdistdk;
      dxd1dk[1] = x[1] * dcdistdk;

      // Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
      Vec4d dxd3dk[2];
      Vec3d dxd3dom[2], dxd3dT[2];
      dxd3dom[0] = dxd1dom[0] + alpha * dxd1dom[1];
      dxd3dom[1] = dxd1dom[1];
      dxd3dT[0] = dxd1dT[0] + alpha * dxd1dT[1];
      dxd3dT[1] = dxd1dT[1];
      dxd3dk[0] = dxd1dk[0] + alpha * dxd1dk[1];
      dxd3dk[1] = dxd1dk[1];

      Vec2d dxd3dalpha(xd1[1], 0);

      // final jacobian
      Jn[0].dom = f[0] * dxd3dom[0];
      Jn[1].dom = f[1] * dxd3dom[1];

      Jn[0].dT = f[0] * dxd3dT[0];
      Jn[1].dT = f[1] * dxd3dT[1];

      Jn[0].dk = f[0] * dxd3dk[0];
      Jn[1].dk = f[1] * dxd3dk[1];

      Jn[0].dalpha = f[0] * dxd3dalpha[0];
      Jn[1].dalpha = 0;  // f[1] * dxd3dalpha[1];

      Jn[0].df = Vec2d(xd3[0], 0);
      Jn[1].df = Vec2d(0, xd3[1]);

      Jn[0].dc = Vec2d(1, 0);
      Jn[1].dc = Vec2d(0, 1);

      // step to jacobian rows for next point
      Jn += 2;
    }
  }
}

void ProjectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
                   cv::InputArray _rvec, cv::InputArray _tvec,
                   const IntrinsicParams& param, cv::OutputArray jacobian) {
  CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
  Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0], 0, param.f[1],
            param.c[1], 0, 0, 1);
  FisheyeProjectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k,
                       param.alpha, jacobian);
}

void RefineExtrinsicsViaOptimization(const Mat& imagePoints,
                                     const Mat& objectPoints, Mat& rvec,
                                     Mat& tvec, Mat& jacobi_extrinsic,
                                     const IntrinsicParams& param,
                                     const double thresh_cond) {
  CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
  CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

  const int MaxIter = 20;
  int iter = 0;
  double updated_ratio = 1;

  while (updated_ratio > 1e-10 && iter < MaxIter) {
    cv::Mat projected_image_points(cv::Size(objectPoints.rows, 2), CV_64FC1);

    // Compute reprojection error and jacobians.
    // # of Jacobian is 15.
    // [fx, fy, cx, cy, k1, k2, k3, k4, rotx, roty, rotz, tx, ty, tz, alpha]
    Mat reproj_diff, jacobians;
    {
      ProjectPoints(objectPoints, projected_image_points, rvec, tvec, param,
                    jacobians);
      reproj_diff = cv::Mat(projected_image_points - imagePoints).reshape(1, 2);
      jacobi_extrinsic = jacobians.colRange(8, 14).clone();
    }

    // Check condition number is smaller than threshold.
    {
      SVD svd(jacobi_extrinsic, SVD::NO_UV);
      double condJJ = svd.w.at<double>(0) / svd.w.at<double>(5);
      if (thresh_cond < condJJ) {
        break;
      }
    }

    // Compute "Innovation Vector" and update extrinsics.
    {
      // Calculate "d_extrinsics" that will compensate reproj_diff.
      Vec6d extrinsics(rvec.at<double>(0), rvec.at<double>(1),
                       rvec.at<double>(2), tvec.at<double>(0),
                       tvec.at<double>(1), tvec.at<double>(2));
      Vec6d d_extrinsics;
      solve(jacobi_extrinsic, reproj_diff.reshape(1, (int)reproj_diff.total()),
            d_extrinsics, DECOMP_SVD + DECOMP_NORMAL);

      // Update parameter (x(t+1) = x(t) - dx)
      extrinsics = extrinsics - d_extrinsics;
      updated_ratio = norm(d_extrinsics) / norm(extrinsics);
      rvec = Mat(Vec3d(extrinsics.val));
      tvec = Mat(Vec3d(extrinsics.val + 3));
    }
    iter++;
  }
}

void CalibrateExtrinsics(const vector<vector<cv::Point3f>>& objectPoints,
                         const vector<vector<cv::Point2f>>& imagePoints,
                         const IntrinsicParams& param, const int check_cond,
                         const double thresh_cond,
                         vector<cv::Vec3d>& board_rotations,
                         vector<cv::Vec3d>& board_translations) {
  CHECK(objectPoints.size() == imagePoints.size());
  CHECK(!objectPoints.empty() && !imagePoints.empty());

  int image_num = imagePoints.size();
  board_rotations.clear();
  board_rotations.resize(image_num);
  board_translations.clear();
  board_translations.resize(image_num);

  for (int image_idx = 0; image_idx < image_num; ++image_idx) {
    Mat board_rot, board_trans, jacobi_extrinsic;

    // Convert vector structure to cv:Mat structure.
    cv::Mat object_points_mat, image_points_mat;
    ConvertVector2dToMat(objectPoints[image_idx], imagePoints[image_idx],
                         object_points_mat, image_points_mat);

    // Compute Extrisics Based on Homography.
    ComputeExtrinsicsBasedOnHomography(image_points_mat, object_points_mat,
                                       param, board_rot, board_trans);

    // Refine Extrinsics calculated in the above function.
    RefineExtrinsicsViaOptimization(image_points_mat, object_points_mat,
                                    board_rot, board_trans, jacobi_extrinsic,
                                    param, thresh_cond);

    // Check condition number for convergence.
    if (check_cond) {
      SVD svd(jacobi_extrinsic, SVD::NO_UV);
      if (svd.w.at<double>(0) / svd.w.at<double>((int)svd.w.total() - 1) >
          thresh_cond) {
        CV_Error(cv::Error::StsInternal,
                 format("CALIB_CHECK_COND - Ill-conditioned "
                        "matrix for input array %d",
                        image_idx));
      }
    }

    board_rot.copyTo(board_rotations[image_idx]);
    board_trans.copyTo(board_translations[image_idx]);
  }
}

void ComputeJacobians(const vector<vector<cv::Point3f>>& objectPoints,
                      const vector<vector<cv::Point2f>>& imagePoints,
                      const IntrinsicParams& param,
                      vector<cv::Vec3d>& board_rotations,
                      vector<cv::Vec3d>& board_translations,
                      const int& check_cond, const double& thresh_cond,
                      Mat& JtJ, Mat& Jtex) {
  int image_num = (int)objectPoints.size();

  // Step 1 : Create approxiomated hessian matrix and error vector.
  // 9 Intrinsic Params + 6 Extrinsic Param * Board Number.
  JtJ = Mat::zeros(9 + 6 * image_num, 9 + 6 * image_num, CV_64FC1);
  Jtex = Mat::zeros(9 + 6 * image_num, 1, CV_64FC1);

  for (int image_idx = 0; image_idx < image_num; ++image_idx) {
    // Step 1 : Convert format.
    cv::Mat object_points_mat, image_points_mat;
    ConvertVector2dToMat(objectPoints[image_idx], imagePoints[image_idx],
                         object_points_mat, image_points_mat);

    // Step 2 : Compute reprojection error and jacobians at projected point.
    Mat jacobians, reproj_err;
    {
      Mat projected_point_mat(cv::Size(2, object_points_mat.rows), CV_64FC1);
      ProjectPoints(object_points_mat, projected_point_mat,
                    Mat(board_rotations[image_idx]),
                    Mat(board_translations[image_idx]), param, jacobians);
      reproj_err = projected_point_mat - image_points_mat;
    }

    // Step3 : Compute jacobians elements of this image.
    Mat J_int, J_ext;
    {
      J_int = Mat(jacobians.rows, 9, CV_64FC1);
      // fx, fy, cx, cy
      jacobians.colRange(0, 4).copyTo(J_int.colRange(0, 4));
      // alpha
      jacobians.col(14).copyTo(J_int.col(4));
      // k1, k2, k3, k4
      jacobians.colRange(4, 8).copyTo(J_int.colRange(5, 9));
      // r1, r2, r3, t1, t2, t3
      J_ext = jacobians.colRange(8, 14).clone();
    }

    // Step4 : Update error vector.
    {
      JtJ(Rect(0, 0, 9, 9)) += J_int.t() * J_int;
      JtJ(Rect(9 + 6 * image_idx, 9 + 6 * image_idx, 6, 6)) = J_ext.t() * J_ext;

      JtJ(Rect(9 + 6 * image_idx, 0, 6, 9)) = J_int.t() * J_ext;
      JtJ(Rect(0, 9 + 6 * image_idx, 9, 6)) =
          JtJ(Rect(9 + 6 * image_idx, 0, 6, 9)).t();

      Jtex.rowRange(0, 9) +=
          J_int.t() * reproj_err.reshape(1, 2 * reproj_err.rows);
      Jtex.rowRange(9 + 6 * image_idx, 9 + 6 * (image_idx + 1)) =
          J_ext.t() * reproj_err.reshape(1, 2 * reproj_err.rows);
    }

    if (check_cond) {
      Mat JJ_kk = J_ext;
      SVD svd(JJ_kk, SVD::NO_UV);
      CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) <
                thresh_cond);
    }
  }

  // Step5 : Shrink matrix if some parameters are not to be estimated.
  std::vector<uchar> idxs(param.isEstimate);
  idxs.insert(idxs.end(), 6 * image_num, 1);
  subMatrix(JtJ, JtJ, idxs, idxs);
  subMatrix(Jtex, Jtex, std::vector<uchar>(1, 1), idxs);
}

void ComputeJacobiansNaively(const vector<vector<cv::Point3f>>& objectPoints,
                             const vector<vector<cv::Point2f>>& imagePoints,
                             const IntrinsicParams& param,
                             vector<cv::Vec3d>& board_rotations,
                             vector<cv::Vec3d>& board_translations,
                             const int& check_cond, const double& thresh_cond,
                             Mat& JtJ, Mat& Jtex) {
  int image_num = (int)objectPoints.size();
  int point_num = (int)objectPoints[0].size();

  // Jacobian
  Mat FullJ = Mat::zeros(cv::Size(9 + 6 * image_num, image_num * point_num * 2),
                         CV_64FC1);
  Mat tmpJ = Mat::zeros(cv::Size(9 + 6 * image_num, image_num * point_num * 2),
                        CV_64FC1);
  Mat ex = Mat::zeros(cv::Size(1, image_num * point_num * 2), CV_64FC1);

  for (int image_idx = 0; image_idx < image_num; ++image_idx) {
    // Convert format.
    cv::Mat object_points_mat, image_points_mat;
    ConvertVector2dToMat(objectPoints[image_idx], imagePoints[image_idx],
                         object_points_mat, image_points_mat);

    // Calculate Reprojection Error.
    Mat jacobians, reproj_err;
    {
      Mat projected_point_mat(cv::Size(2, object_points_mat.rows), CV_64FC1);
      ProjectPoints(object_points_mat, projected_point_mat,
                    Mat(board_rotations[image_idx]),
                    Mat(board_translations[image_idx]), param, jacobians);
      reproj_err = projected_point_mat - image_points_mat;
    }

    // Create Intrinsic & Extrinsic Part of Jacobian.
    Mat J_int, J_ext;
    {
      J_int = Mat(jacobians.rows, 9, CV_64FC1);
      // fx, fy, cx, cy
      jacobians.colRange(0, 4).copyTo(J_int.colRange(0, 4));
      // alpha
      jacobians.col(14).copyTo(J_int.col(4));
      // k1, k2, k3, k4
      jacobians.colRange(4, 8).copyTo(J_int.colRange(5, 9));
      // r1, r2, r3, t1, t2, t3
      J_ext = jacobians.colRange(8, 14).clone();
    }

    // Create Jacobian and Error Vector
    {
      tmpJ = 0;
      tmpJ(Rect(0, point_num * image_idx * 2, 9, point_num * 2)) += J_int;
      tmpJ(Rect(9 + 6 * image_idx, image_idx * point_num * 2, 6,
                point_num * 2)) += J_ext;
      FullJ += tmpJ;
      ex(Rect(0, point_num * image_idx * 2, 1, point_num * 2)) +=
          reproj_err.reshape(1, 2 * reproj_err.rows);
    }

    if (check_cond) {
      Mat JJ_kk = J_ext;
      SVD svd(JJ_kk, SVD::NO_UV);
      CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) <
                thresh_cond);
    }
  }

  JtJ = FullJ.t() * FullJ;
  Jtex = FullJ.t() * ex;

  std::vector<uchar> idxs(param.isEstimate);
  idxs.insert(idxs.end(), 6 * image_num, 1);
  subMatrix(JtJ, JtJ, idxs, idxs);
  subMatrix(Jtex, Jtex, std::vector<uchar>(1, 1), idxs);
}

void InitializeParamsBasedOnFlagSetting(const int flags, const cv::Matx33d& K,
                                        const std::vector<double>& D,
                                        const cv::Size& image_size,
                                        IntrinsicParams& params,
                                        int& check_cond,
                                        int& recompute_extrinsic) {
  // Load Calibration Setting from flags.
  params.isEstimate[0] = 1;
  params.isEstimate[1] = 1;
  params.isEstimate[2] = flags & cv::fisheye::CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
  params.isEstimate[3] = flags & cv::fisheye::CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
  params.isEstimate[4] = flags & cv::fisheye::CALIB_FIX_SKEW ? 0 : 1;
  params.isEstimate[5] = flags & cv::fisheye::CALIB_FIX_K1 ? 0 : 1;
  params.isEstimate[6] = flags & cv::fisheye::CALIB_FIX_K2 ? 0 : 1;
  params.isEstimate[7] = flags & cv::fisheye::CALIB_FIX_K3 ? 0 : 1;
  params.isEstimate[8] = flags & cv::fisheye::CALIB_FIX_K4 ? 0 : 1;

  // Step 1. Initialization.
  if (flags & CALIB_USE_INTRINSIC_GUESS) {
    CHECK(D.size() == 4)
        << "Distortion coefficient for fisheye lens must be 4.";
    params.Init(
        Vec2d(K(0, 0), K(1, 1)), Vec2d(K(0, 2), K(1, 2)),
        Vec4d(flags & CALIB_FIX_K1 ? 0 : D[0], flags & CALIB_FIX_K2 ? 0 : D[1],
              flags & CALIB_FIX_K3 ? 0 : D[2], flags & CALIB_FIX_K4 ? 0 : D[3]),
        K(0, 1) / K(0, 0));
  } else {
    params.Init(
        Vec2d(max(image_size.width, image_size.height) / CV_PI,
              max(image_size.width, image_size.height) / CV_PI),
        Vec2d(image_size.width / 2.0 - 0.5, image_size.height / 2.0 - 0.5));
  }

  recompute_extrinsic = flags & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? 1 : 0;
  check_cond = flags & cv::fisheye::CALIB_CHECK_COND ? 1 : 0;
}

bool MinimizeReprojectionError(const cv::TermCriteria& criteria,
                               const vector<vector<cv::Point3f>>& objectPoints,
                               const vector<vector<cv::Point2f>>& imagePoints,
                               const int check_cond, const double thresh_cond,
                               const int recompute_extrinsic,
                               IntrinsicParams& final_param,
                               std::vector<Vec3d>& board_rotations,
                               std::vector<Vec3d>& board_translations) {
  double updated_ratio = 1;
  cv::Vec2d err_std;
  for (int iter = 0; iter <= std::numeric_limits<int>::max(); ++iter) {
    // Step 1 : Condition check for loop termination.
    if ((criteria.type == 1 && iter >= criteria.maxCount) ||
        (criteria.type == 2 && updated_ratio <= criteria.epsilon) ||
        (criteria.type == 3 &&
         (updated_ratio <= criteria.epsilon || iter >= criteria.maxCount))) {
      break;
    }

    // Step 2 :
    // Compute jacobian and gradient in parameter space. (Gauss Newton 1)
    Mat d_param;
    {
      Mat JtJ, Jtex;
#if 1
      ComputeJacobians(objectPoints, imagePoints, final_param, board_rotations,
                       board_translations, check_cond, thresh_cond, JtJ, Jtex);
#else
      ComputeJacobiansNaively(objectPoints, imagePoints, final_param,
                              board_rotations, board_translations, check_cond,
                              thresh_cond, JtJ, Jtex);
#endif
      solve(JtJ, -Jtex, d_param);
    }

    // Step 3 :
    // Update parameter by d_param * alpha_smooth2. (Gauss Newton 2)
    {
      const double alpha_smooth = 0.4;
      double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, iter + 1.0);
      IntrinsicParams temp_param = final_param + alpha_smooth2 * d_param;
      updated_ratio = norm(Vec4d(temp_param.f[0], temp_param.f[1],
                                 temp_param.c[0], temp_param.c[1]) -
                           Vec4d(final_param.f[0], final_param.f[1],
                                 final_param.c[0], final_param.c[1])) /
                      norm(Vec4d(temp_param.f[0], temp_param.f[1],
                                 temp_param.c[0], temp_param.c[1]));
      final_param = temp_param;
    }

    // Step 4 :
    // Recomputing extrinsics based on the updated parameter sets.
    if (recompute_extrinsic) {
      CalibrateExtrinsics(objectPoints, imagePoints, final_param, check_cond,
                          thresh_cond, board_rotations, board_translations);
    }
  }

  return true;
}

void EstimateUncertainties(const vector<vector<cv::Point3f>>& objectPoints,
                           const vector<vector<cv::Point2f>>& imagePoints,
                           const IntrinsicParams& params,
                           vector<cv::Vec3d> board_rotations,
                           vector<cv::Vec3d> board_translations, Vec2d& std_err,
                           double& rms) {
  CV_Assert(!objectPoints.empty() && !imagePoints.empty());

  // Create error vector of width 2.
  Mat reproj_err_vector;
  {
    int total_ex = 0;
    for (int image_idx = 0; image_idx < (int)objectPoints.size(); ++image_idx) {
      total_ex += objectPoints[image_idx].size() * 3;
    }
    reproj_err_vector = Mat(total_ex, 1, CV_64FC2);
  }

  // Project point based on the calibrated model and fill error vector.
  {
    int image_num = objectPoints.size();
    for (int image_idx = 0; image_idx < image_num; ++image_idx) {
      cv::Mat object_points_mat, image_points_mat;
      ConvertVector2dToMat(objectPoints[image_idx], imagePoints[image_idx],
                           object_points_mat, image_points_mat);

      cv::Mat projected_point_mat(cv::Size(2, image_points_mat.rows), CV_64FC1);
      ProjectPoints(object_points_mat, projected_point_mat,
                    Mat(board_rotations[image_idx]),
                    Mat(board_translations[image_idx]), params, noArray());
      Mat reproj_err = image_points_mat - projected_point_mat;

      int point_num = objectPoints[image_idx].size();
      reproj_err.copyTo(reproj_err_vector.rowRange(
          point_num * image_idx, point_num * (image_idx + 1)));
    }
  }

  // Compute statistics.
  {
    meanStdDev(reproj_err_vector, noArray(), std_err);
    std_err *= sqrt((double)reproj_err_vector.total() /
                    ((double)reproj_err_vector.total() - 1.0));
    rms = sqrt(norm(reproj_err_vector, NORM_L2SQR) / reproj_err_vector.total());
  }
}

void ConvertFormat(const IntrinsicParams& finalParam, cv::Matx33d& K,
                   std::vector<double>& D) {
  K = Matx33d(finalParam.f[0], finalParam.f[0] * finalParam.alpha,
              finalParam.c[0], 0, finalParam.f[1], finalParam.c[1], 0, 0, 1);

  D.clear();
  finalParam.k(0);
  for (int idx = 0; idx < finalParam.k.rows; idx++) {
    D.push_back(finalParam.k(idx));
  }
}

double CalibrateFisheye(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const cv::Size& image_size, const int flags,
    const cv::TermCriteria criteria, cv::Matx33d& K, std::vector<double>& D,
    std::vector<cv::Vec3d>& board_rotations,
    std::vector<cv::Vec3d>& board_translations) {
  IntrinsicParams final_param;

  // Step 1. Initailization of the parameter
  int check_cond, recompute_extrinsic;
  InitializeParamsBasedOnFlagSetting(flags, K, D, image_size, final_param,
                                     check_cond, recompute_extrinsic);

  // Step 2. Calculate Homography and Extrinsics
  const double thresh_cond = 1e6;
  CalibrateExtrinsics(objectPoints, imagePoints, final_param, check_cond,
                      thresh_cond, board_rotations, board_translations);

  // Step 3. Minimize Reprojection Error.
  cv::Vec2d err_std;
  MinimizeReprojectionError(criteria, objectPoints, imagePoints, check_cond,
                            thresh_cond, recompute_extrinsic, final_param,
                            board_rotations, board_translations);

  // Step 4. Calib Result Summary.
  double rms;
  EstimateUncertainties(objectPoints, imagePoints, final_param, board_rotations,
                        board_translations, err_std, rms);

  // Step 5. Format Conversion.
  ConvertFormat(final_param, K, D);

  return rms;
}
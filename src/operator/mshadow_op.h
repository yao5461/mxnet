/*!
 * Copyright (c) 2015 by Contributors
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu, Ming Zhang
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_

#include <mxnet/base.h>

namespace mxnet {
namespace op {
namespace mshadow_op {
/*! \brief identity Operation */
struct identity {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a;
  }
};

struct identity_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f;
  }
};


struct negation {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return -a;
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? a : 0.0f;
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0.0f ? a : a * b;
  }
};

struct xelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0.0f ? 1.0f : b;
  }
};

struct tanh {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return tanhf( a );
  }
};

struct tanh_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f - a * a;
  }
};

struct exp {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return expf(a);
  }
};

struct log {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return logf(a);
  }
};

struct log_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / a;
  }
};

struct square {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a;
  }
};

struct square_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 2.0f * a;
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

/*! \brief used for generate element of power */
struct power {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return powf( a, b );
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return sqrt(a);
  }
};

struct square_root_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 0.5f / a;
  }
};


/*!\ \brief used for generate element smooth l1 */
struct smooth_l1 {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    real_t ret = 0.0f;
    real_t abs_a = fabs(a);
    if (abs_a < 1.0f) {
      ret = 0.5f * a * a;
    }
    else {
      ret = abs_a - 0.5f;
    }
    return ret;
  }
};

struct smooth_l1_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    real_t ret = 0.0f;
    real_t abs_a = fabs(a);
    if (abs_a < 1.0f) {
      ret = a;
    }
    else {
      ret = a < 0.f ? -1.5f : 0.5f;
    }
    return ret;
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_

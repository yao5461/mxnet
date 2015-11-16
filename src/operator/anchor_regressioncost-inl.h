/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_regressioncost-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_OPERATOR_ANCHOR_REGRESSIONCOST_INL_H_
#define MXNET_OPERATOR_ANCHOR_REGRESSIONCOST_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#define MIN_NUM 1e-37f

namespace mxnet {
namespace op {

namespace anchor_regcost_enum {
enum AnchorRegCostOpInputs {kData, kLabel, kCoordLabel, kBBsLabel, kAnchorInfoLabel};
enum AnchorRegCostOpOutputs {kOut};
enum AnchorRegCostOpResource {kTempSpace};
};


struct AnchorRegCostParam : public dmlc::Parameter<AnchorRegCostParam> {
  // use int for enumeration
  uint32_t anchornum;
  DMLC_DECLARE_PARAMETER(AnchorRegCostParam) {
    DMLC_DECLARE_FIELD(anchornum)
    .set_default(0)
    .describe("The Anchor Number.");
  }
};


template<typename xpu>
class AnchorRegCostOp : public Operator {
 public:
  explicit AnchorRegCostOp(AnchorRegCostParam p) {
    CHECK_NE(p.anchornum, 0) << "anchornum can not be equal 0.";
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    index_t anchornum = param_.anchornum;
    
    TBlob data_in = in_data[anchor_regcost_enum::kData];
    TBlob label = in_data[anchor_regcost_enum::kLabel];
    TBlob coordlabel = in_data[anchor_regcost_enum::kCoordLabel];
    TBlob bbslabel = in_data[anchor_regcost_enum::kBBsLabel];
    TBlob infolabel = in_data[anchor_regcost_enum::kAnchorInfoLabel];
    TBlob data_out = out_data[anchor_regcost_enum::kOut];
    
    TShape shape_in = data_in.shape_;

    Tensor<xpu, 4> tdata_in = data_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tlabel = label.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tcoordlabel = coordlabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> tbbslabel = bbslabel.get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> tinfolabel = infolabel.get<xpu, 2, real_t>(s);
    Tensor<xpu, 4> tdata_out = data_out.get<xpu, 4, real_t>(s);
    for (index_t bi = 0; bi < shape_in[0]; bi++) {
      const Tensor<xpu, 1> &onebb = tbbslabel[bi];
      const Tensor<xpu, 3> &coords = tcoordlabel[bi];
      for (index_t ai = 0; ai < anchornum; ai++) {
        const Tensor<xpu, 2> &onelabel = tlabel[bi][ai];
        const Tensor<xpu, 3> &onedatas = tdata_in[bi].Slice(ai * 4, (ai + 1) * 4);
        const Tensor<xpu, 3> &oneouts = tdata_out[bi].Slice(ai * 4, (ai + 1) * 4);
        const Tensor<xpu, 1> &oneinfo = tinfolabel[ai];
        for (index_t di = 0; di < 2; di++) {
          const Tensor<xpu, 2> &onedata = onedatas[di];
          const Tensor<xpu, 2> &onecoord = coords[di];
          oneouts[di] = F<mshadow_op::square>(onedata -
                       ((onebb[di] - onecoord) / oneinfo[di])) * onelabel / 2;
        }
        const Tensor<xpu, 3> &onedata2 = onedatas.Slice(2, 4);
        const Tensor<xpu, 1> &partbb = onebb.Slice(2, 4);
        for (index_t di = 0; di < 2; di++) {
          real_t t_star = logf(partbb[di] / oneinfo[di] + MIN_NUM);
          oneouts[di] = F<mshadow_op::square>(onedata2[di] - t_star) * onelabel / 2;
        }
      }
    }
    
  }

  virtual void Backward(const OpContext &ctx,
                       const std::vector<TBlob> &out_grad,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &in_grad,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    index_t anchornum = param_.anchornum;

    TBlob data_in = in_data[anchor_regcost_enum::kData];
    TBlob label = in_data[anchor_regcost_enum::kLabel];
    TBlob coordlabel = in_data[anchor_regcost_enum::kCoordLabel];
    TBlob bbslabel = in_data[anchor_regcost_enum::kBBsLabel];
    TBlob infolabel = in_data[anchor_regcost_enum::kAnchorInfoLabel];
    TBlob grad_in = in_grad[anchor_regcost_enum::kOut];
    
    TShape shape_in = data_in.shape_;
    
    Tensor<xpu, 4> tdata_in = data_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tlabel = label.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tcoordlabel = coordlabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> tbbslabel = bbslabel.get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> tinfolabel = infolabel.get<xpu, 2, real_t>(s);
    Tensor<xpu, 4> tgrad_in = grad_in.get<xpu, 4, real_t>(s);
    
    for (index_t bi = 0; bi < shape_in[0]; bi++) {
      const Tensor<xpu, 1> &onebb = tbbslabel[bi];
      const Tensor<xpu, 3> &coords = tcoordlabel[bi];
      for (index_t ai = 0; ai < anchornum; ai++) {
        const Tensor<xpu, 2> &onelabel = tlabel[bi][ai];
        const Tensor<xpu, 3> &onedatas = tdata_in[bi].Slice(ai * 4, (ai + 1) * 4);
        const Tensor<xpu, 3> &onegrads = tgrad_in[bi].Slice(ai * 4, (ai + 1) * 4);
        const Tensor<xpu, 1> &oneinfo = tinfolabel[ai];
        for (index_t di = 0; di < 2; di++) {
          const Tensor<xpu, 2> &onedata = onedatas[di];
          const Tensor<xpu, 2> &onecoord = coords[di];
          onegrads[di] = (onedata - ((onebb[di] - onecoord) / oneinfo[di])) * onelabel;
        }
        const Tensor<xpu, 3> &onedata2 = onedatas.Slice(2, 4);
        const Tensor<xpu, 1> &partbb = onebb.Slice(2, 4);
        for (index_t di = 0; di < 2; di++) {
          real_t t_star = logf(partbb[di] / oneinfo[di] + MIN_NUM);
          onegrads[di] = (onedata2[di] - t_star) * onelabel;
        }
      }
    }
    
  }

  AnchorRegCostParam param_;
};


template<typename xpu>
Operator* CreateOp(AnchorRegCostParam param);


#if DMLC_USE_CXX11
class AnchorRegCostProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "coordlabel", "bbslabel", "infolabel"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 5);

    TShape &datashape = (*in_shape)[anchor_regcost_enum::kData];
    TShape &labelshape = (*in_shape)[anchor_regcost_enum::kLabel];
    TShape &coordshape = (*in_shape)[anchor_regcost_enum::kCoordLabel];
    TShape &bbshape = (*in_shape)[anchor_regcost_enum::kBBsLabel];
    TShape &infoshape = (*in_shape)[anchor_regcost_enum::kAnchorInfoLabel];

    labelshape = Shape4(datashape[0], param_.anchornum, datashape[2], datashape[3]);
    coordshape = Shape4(datashape[0], 2, datashape[2], datashape[3]);
    bbshape = Shape2(datashape[0], 4);
    infoshape = Shape2(param_.anchornum, 2);

    out_shape->clear();
    out_shape->push_back(datashape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AnchorRegCostProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "AnchorRegCost";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_data[anchor_regcost_enum::kOut],
            in_data[anchor_regcost_enum::kData],
            in_data[anchor_regcost_enum::kLabel],
            in_data[anchor_regcost_enum::kCoordLabel],
            in_data[anchor_regcost_enum::kBBsLabel],
            in_data[anchor_regcost_enum::kAnchorInfoLabel],
            };
  };


  Operator* CreateOperator(Context ctx) const override;

 private:
  AnchorRegCostParam param_;
};  // class AnchorRegCostProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ANCHOR_REGRESSIONCOST_INL_H_



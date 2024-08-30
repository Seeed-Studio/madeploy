#ifndef _MA_MODEL_BASE_H_
#define _MA_MODEL_BASE_H_

#include <functional>

#include "core/ma_common.h"

#include "core/engine/ma_engine.h"
namespace ma {

typedef std::function<void(void* ctx)> ModelCallback;

using namespace ma::engine;
class Model {
private:
    ma_perf_t perf_;
    ModelCallback p_preprocess_done_;
    ModelCallback p_postprocess_done_;
    ModelCallback p_run_done_;
    void* p_user_ctx_;
    ma_model_type_t m_type_;

protected:
    Engine* p_engine_;
    const char* p_name_;
    virtual ma_err_t preprocess()  = 0;
    virtual ma_err_t postprocess() = 0;
    ma_err_t underlyingRun();

public:
    Model(Engine* engine, const char* name, ma_model_type_t type);
    virtual ~Model();
    const ma_perf_t getPerf() const;
    const char* getName() const;
    ma_model_type_t getType() const;
    virtual ma_err_t setConfig(ma_model_cfg_opt_t opt, ...) = 0;
    virtual ma_err_t getConfig(ma_model_cfg_opt_t opt, ...) = 0;
    void setPreprocessDone(ModelCallback cb);
    void setPostprocessDone(ModelCallback cb);
    void setRunDone(ModelCallback cb);
    void setUserCtx(void* ctx);
};
}  // namespace ma

#endif /* _MA_ALGO_H_ */

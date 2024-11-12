#ifndef _MA_CONFIG_H_
#define _MA_CONFIG_H_

#define CONFIG_MA_DEBUG_LEVEL         6
#define CONFIG_MA_DEBUG_MORE_INFO     1

#define CONFIG_MA_ENGINE_TFLITE       1
#define CONFIG_MA_TFLITE_OP_ALL       1
#define CONFIG_USE_ENGINE_TENSOR_NAME 1

#define CONFIG_MA_FILESYSTEM          1
#define CONFIG_MA_FILESYSTEM_POSIX    

#ifndef CONFIG_MA_TRANSPORT_MQTT
#define CONFIG_MA_TRANSPORT_MQTT 1
#endif

#define ma_malloc                       malloc
#define ma_calloc                       calloc
#define ma_realloc                      realloc
#define ma_free                         free
#define ma_printf                       printf
#define ma_abort                        abort
#define ma_reset                        abort


#endif /* _EL_CONFIG_H_ */
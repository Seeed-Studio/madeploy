#ifndef _MA_CONFIG_INTERNAL_H_
#define _MA_CONFIG_INTERNAL_H_


#include "ma_config.h"

/* debug config check */
#ifndef CONFIG_MA_DEBUG_LEVEL
// Level:
//      0: no debug
//      1: print error
//      2: print warning
//      3: print info
//      4: print debug
//      5: print verbose
#define MA_DEBUG_LEVEL 3
#else
#define MA_DEBUG_LEVEL CONFIG_MA_DEBUG_LEVEL
#endif
#ifndef CONFIG_MA_DEBUG_MORE_INFO
#define MA_USE_DEBUG_MORE_INFO 1
#else
#define MA_USE_DEBUG_MORE_INFO CONFIG_MA_DEBUG_MORE_INFO
#endif


/* assert config check */
#ifndef CONFIG_MA_ASSERT
#if MA_DEBUG_LEVEL > 0
#define MA_USE_ASSERT 1
#endif
#else
#define MA_USE_ASSERT CONFIG_MA_ASSERT
#endif

#ifndef CONFIG_MA_STATIC_ASSERT
#define MA_USE_STATIC_ASSERT 1
#else
#define MA_USE_STATIC_ASSERT CONFIG_MA_STATIC_ASSERT
#endif

/* filesystem config check */
#ifdef CONFIG_MA_FILESYSTEM
#define MA_USE_FILESYSTEM 1
#endif

#ifdef CONFIG_MA_FILESYSTEM_POSIX
#define MA_USE_FILESYSTEM_POSIX 1
#endif

/* porting config check */
#ifndef CONFIG_MA_PORTING_POSIX
#if defined(__unix__) || defined(__APPLE__)
#define MA_PORTING_POSIX 1
#ifndef MA_USE_FILESYSTEM
#define MA_USE_FILESYSTEM 1
#endif
#ifndef MA_USE_FILESYSTEM_POSIX
#define MA_USE_FILESYSTEM_POSIX 1
#endif
#else
#define MA_PORTING_POSIX 0
#endif
#endif
#ifndef CONFIG_MA_TIMER_PRIO
#define MA_TIMER_PRIO 30
#else
#if CONFIG_MA_TIMER_PRIO <= 0
#error "MA_TIMER_PRIO must be greater than 0"
#endif
#define MA_TIMER_PRIO CONFIG_MA_TIMER_PRIO
#endif

/* engine config check */

#if CONFIG_MA_ENGINE_TFLITE + CONFIG_MA_ENGINE_CVINN > 1
#error "Only one engine can be enabled"
#endif

#ifdef CONFIG_MA_ENGINE_TENSOR_NAME
#define MA_USE_ENGINE_TENSOR_NAME 1
#endif

#ifdef CONFIG_MA_ENGINE_TFLITE
#define MA_USE_ENGINE_TFLITE 1
#if MA_USE_ENGINE_TENSOR_NAME
#error "TensorFlow Lite engine does not support tensor name"
#endif
#ifndef CONFIG_MA_ENGINE_TFLITE_TENSOE_ARENA_SIZE
#define MA_ENGINE_TFLITE_TENSOE_ARENA_SIZE 1024 * 1024
#else
#define MA_ENGINE_TFLITE_TENSOE_ARENA_SIZE CONFIG_MA_ENGINE_TFLITE_TENSOE_ARENA_SIZE
#endif
#if CONFIG_MA_TFLITE_OP_ALL
#define MA_TFLITE_OP_ALL 1
#else
#ifdef CONFIG_MA_TFLITE_OP_ABS
#define MA_TFLITE_OP_ABS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ADD
#define MA_TFLITE_OP_ADD 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ADDN
#define MA_TFLITE_OP_ADDN 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ARGMAX
#define MA_TFLITE_OP_ARGMAX 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ARGMIN
#define MA_TFLITE_OP_ARGMIN 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ASSIGN_VARIABLE
#define MA_TFLITE_OP_ASSIGN_VARIABLE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_AVERAGE_POOL_2D
#define MA_TFLITE_OP_AVERAGE_POOL_2D 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_BATCH_TO_SPACE_ND
#define MA_TFLITE_OP_BATCH_TO_SPACE_ND 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_BROADCAST_ARGS
#define MA_TFLITE_OP_BROADCAST_ARGS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_BROADCAST_TO
#define MA_TFLITE_OP_BROADCAST_TO 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CALL_ONCE
#define MA_TFLITE_OP_CALL_ONCE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CAST
#define MA_TFLITE_OP_CAST 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CEIL
#define MA_TFLITE_OP_CEIL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CIRULAR_BUFFER
#define MA_TFLITE_OP_CIRULAR_BUFFER 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CONCATENATION
#define MA_TFLITE_OP_CONCATENATION 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CONV_2D
#define MA_TFLITE_OP_CONV_2D 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_COS
#define MA_TFLITE_OP_COS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_CUM_SUM
#define MA_TFLITE_OP_CUM_SUM 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_DEPTH_TO_SPACE
#define MA_TFLITE_OP_DEPTH_TO_SPACE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_DEPTHWISE_CONV_2D
#define MA_TFLITE_OP_DEPTHWISE_CONV_2D 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_DEQUANTIZE
#define MA_TFLITE_OP_DEQUANTIZE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_DETECTION_POSTPROCESS
#define MA_TFLITE_OP_DETECTION_POSTPROCESS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_DIV
#define MA_TFLITE_OP_DIV 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MAU
#define MA_TFLITE_OP_MAU 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_EQUAL
#define MA_TFLITE_OP_EQUAL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ETHOS_U
#define MA_TFLITE_OP_ETHOS_U 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_EXP
#define MA_TFLITE_OP_EXP 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_EXPAND_DIMS
#define MA_TFLITE_OP_EXPAND_DIMS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_FILL
#define MA_TFLITE_OP_FILL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_FLOOR
#define MA_TFLITE_OP_FLOOR 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_FLOOR_DIV
#define MA_TFLITE_OP_FLOOR_DIV 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_FLOOR_MOD
#define MA_TFLITE_OP_FLOOR_MOD 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_FULLY_CONNECTED
#define MA_TFLITE_OP_FULLY_CONNECTED 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_GATHER
#define MA_TFLITE_OP_GATHER 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_GATHER_ND
#define MA_TFLITE_OP_GATHER_ND 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_GREATER
#define MA_TFLITE_OP_GREATER 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_GREATER_EQUAL
#define MA_TFLITE_OP_GREATER_EQUAL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_HARD_SWISH
#define MA_TFLITE_OP_HARD_SWISH 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_IF
#define MA_TFLITE_OP_IF 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_L2_NORMALIZATION
#define MA_TFLITE_OP_L2_NORMALIZATION 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_L2_POOL_2D
#define MA_TFLITE_OP_L2_POOL_2D 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LEAKY_RMAU
#define MA_TFLITE_OP_LEAKY_RMAU 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LESS
#define MA_TFLITE_OP_LESS 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LESS_EQUAL
#define MA_TFLITE_OP_LESS_EQUAL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOG
#define MA_TFLITE_OP_LOG 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOGICAL_AND
#define MA_TFLITE_OP_LOGICAL_AND 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOGICAL_NOT
#define MA_TFLITE_OP_LOGICAL_NOT 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOGICAL_OR
#define MA_TFLITE_OP_LOGICAL_OR 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOGISTIC
#define MA_TFLITE_OP_LOGISTIC 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_LOG_SOFTMAX
#define MA_TFLITE_OP_LOG_SOFTMAX 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MAX_POOL_2D
#define MA_TFLITE_OP_MAX_POOL_2D 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MAXIMUM
#define MA_TFLITE_OP_MAXIMUM 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MEAN
#define MA_TFLITE_OP_MEAN 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MINIMUM
#define MA_TFLITE_OP_MINIMUM 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MIRROR_PAD
#define MA_TFLITE_OP_MIRROR_PAD 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_MUL
#define MA_TFLITE_OP_MUL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_NEG
#define MA_TFLITE_OP_NEG 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_NOT_EQUAL
#define MA_TFLITE_OP_NOT_EQUAL 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_PACK
#define MA_TFLITE_OP_PACK 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_PAD
#define MA_TFLITE_OP_PAD 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_PADV2
#define MA_TFLITE_OP_PADV2 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_PRMAU
#define MA_TFLITE_OP_PRMAU 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_QUANTIZE
#define MA_TFLITE_OP_QUANTIZE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_READ_VARIABLE
#define MA_TFLITE_OP_READ_VARIABLE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_REDUCE_ANY
#define MA_TFLITE_OP_REDUCE_ANY 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RMAU
#define MA_TFLITE_OP_RMAU 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RMAU6
#define MA_TFLITE_OP_RMAU6 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RESHAPE
#define MA_TFLITE_OP_RESHAPE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RESIZE_BILINEAR
#define MA_TFLITE_OP_RESIZE_BILINEAR 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RESIZE_NEAREST_NEIGHBOR
#define MA_TFLITE_OP_RESIZE_NEAREST_NEIGHBOR 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ROUND
#define MA_TFLITE_OP_ROUND 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_RSQRT
#define MA_TFLITE_OP_RSQRT 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SMAECT_V2
#define MA_TFLITE_OP_SMAECT_V2 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SHAPE
#define MA_TFLITE_OP_SHAPE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SIN
#define MA_TFLITE_OP_SIN 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SLICE
#define MA_TFLITE_OP_SLICE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SOFTMAX
#define MA_TFLITE_OP_SOFTMAX 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SPACE_TO_BATCH_ND
#define MA_TFLITE_OP_SPACE_TO_BATCH_ND 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SPACE_TO_DEPTH
#define MA_TFLITE_OP_SPACE_TO_DEPTH 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SPLIT
#define MA_TFLITE_OP_SPLIT 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SPLIT_V
#define MA_TFLITE_OP_SPLIT_V 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SQRT
#define MA_TFLITE_OP_SQRT 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SQUARE
#define MA_TFLITE_OP_SQUARE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SQUARED_DIFFERENCE
#define MA_TFLITE_OP_SQUARED_DIFFERENCE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SQUEEZE
#define MA_TFLITE_OP_SQUEEZE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_STRIDED_SLICE
#define MA_TFLITE_OP_STRIDED_SLICE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SUB
#define MA_TFLITE_OP_SUB 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SUM
#define MA_TFLITE_OP_SUM 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_SVDF
#define MA_TFLITE_SVDF 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_TANH
#define MA_TFLITE_OP_TANH 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_TRANSPOSE
#define MA_TFLITE_OP_TRANSPOSE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_TRANSPOSE_CONV
#define MA_TFLITE_OP_TRANSPOSE_CONV 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_UNIDIRECTIONAL_SEQUENCE_LSTM
#define MA_TFLITE_OP_UNIDIRECTIONAL_SEQUENCE_LSTM 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_UNPACK
#define MA_TFLITE_OP_UNPACK 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_VARHANDLE
#define MA_TFLITE_OP_VARHANDLE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_WHILE
#define MA_TFLITE_OP_WHILE 1
#endif
#ifdef CONFIG_MA_TFLITE_OP_ZEROS_LIKE
#define MA_TFLITE_OP_ZEROS_LIKE 1
#endif
#endif
#endif

#ifndef CONFIG_MA_ENGINE_SHAPE_MAX_DIM
#define MA_ENGINE_SHAPE_MAX_DIM 6
#endif

#endif

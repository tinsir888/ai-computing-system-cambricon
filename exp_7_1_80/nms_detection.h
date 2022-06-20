/*copyright (C) [2020] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef _NMS_DETECTION_H_
  #define _NMS_DETECTION_H_
#endif

#define NMS_DT half
#define NMS_SIZE 64
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y

enum Addr { NRAM, SRAM, GDRAM };
enum SplitMode { NMS_BLOCK = 1, NMS_U1 = 4 };

// max(x, y) ~ max(x - y, 0) + y
__mlu_func__ void __svmax_relu(NMS_DT *dst, NMS_DT *src0, NMS_DT *src1, int len) {
  __bang_sub(dst, src0, src1, len);
  __bang_active_relu(dst, dst, len);
  __bang_add(dst, dst, src1, len);
}

// min(x, y) ~ y - max(y - x, 0)
__mlu_func__ void __svmin_relu(NMS_DT *dst, NMS_DT *src0, NMS_DT *src1, int len) {
  __bang_sub(dst, src1, src0, len);
  __bang_active_relu(dst, dst, len);
  __bang_sub(dst, src1, dst, len);
}
/* 
 实现NMS，支持NRAM
  src == NRAM
  split_mode == NMS_BLOCK
  save_method == 1
  MODE == 1
 */

__mlu_func__ void nms_detection(
         int& output_box_num,     
         NMS_DT* output_data,     
         Addr dst,                  
         NMS_DT* input_data_score, 
         NMS_DT* input_data_box,
         Addr src,
         NMS_DT *buffer,
         int buffer_size,
         NMS_DT* sram,
         SplitMode split_mode,
         int input_box_num,
         int input_stride,
         int output_stride,
         int keepNum,
         NMS_DT thresh_iou,
         NMS_DT thresh_score,
         int save_method){
  
  int core_limit = split_mode;
  int32_t* loop_end_flag;
  int nram_save_limit_count = 0;
  nram_save_limit_count = dst == NRAM ? 0 : 256;

  int MODE;

  NMS_DT* input_score_ptr;
  NMS_DT* input_x1_ptr;
  NMS_DT* input_y1_ptr;
  NMS_DT* input_x2_ptr;
  NMS_DT* input_y2_ptr;
  input_score_ptr = input_data_score;
  input_x1_ptr = input_data_box;
  input_y1_ptr = input_x1_ptr + input_stride;
  input_x2_ptr = input_y1_ptr + input_stride;
  input_y2_ptr = input_x2_ptr + input_stride;

  NMS_DT* score;
  NMS_DT* inter_x1;
  NMS_DT* inter_y1;
  NMS_DT* inter_x2;
  NMS_DT* inter_y2;
  NMS_DT* max_box;
  NMS_DT* nram_save;

  inter_x1 = buffer;
  inter_y1 = inter_x1 + input_box_num;
  inter_x2 = inter_y1 + input_box_num;
  inter_y2 = inter_x2 + input_box_num;

  score  = input_data_score;
  max_box = inter_y2 + input_box_num;
  nram_save = max_box + 64;

  int input_offset = 0;
  int nram_save_count = 0;
  int seg_len = input_box_num;

  NMS_DT max_area;
  unsigned int max_index;

  NMS_DT* save_ptr;
  int save_offset = 0;
  int save_str_num = 0;

  mluMemcpyDirection_t store_dir;

  output_box_num = 0;
  for (int keep = 0; keep < keepNum; keep++) {
    max_index = 0;
    max_area = 0;
    max_box[0] = 0;

    __bang_max(inter_x1, score, input_box_num);
    if (inter_x1[0] > max_box[0]){
      max_box[0] = inter_x1[0];
      max_index = ((unsigned short*)inter_x1)[1] + input_offset;
    }

    max_box[1] = input_x1_ptr[max_index];
    max_box[2] = input_y1_ptr[max_index];
    max_box[3] = input_x2_ptr[max_index];
    max_box[4] = input_y2_ptr[max_index];
  
    max_area = (max_box[4]-max_box[2]) * (max_box[3]-max_box[1]);
    
    score[max_index] = 0;
  
    if (dst != NRAM && output_box_num != 0) {
      store_dir = dst == SRAM ? NRAM2SRAM : NRAM2GDRAM;

      if ((nram_save_count == nram_save_limit_count) || (max_box[0] <= thresh_score)) {
        __memcpy(output_data,
            nram_save,
            nram_save_count * sizeof(NMS_DT),
            store_dir,
            output_stride * sizeof(NMS_DT),
            nram_save_limit_count * sizeof(NMS_DT),
            4);
        output_data += nram_save_count;    
        nram_save_count = 0;
      } 
    } 

    if (max_box[0] <= thresh_score)
      break;

    if (dst == NRAM) {
      save_ptr = output_data;
      save_offset = output_box_num;
      save_str_num = input_box_num;
    } else {
      save_ptr = nram_save;
      save_offset = nram_save_count;
      save_str_num = nram_save_limit_count;
    }
    //void __memcpy (void* dst, void* src, int size, mluMemcpyDirection_t dir, 
    //              int dst_stride, int src_stride, int count, int id_dst_cluster);
    __memcpy(save_ptr + save_offset, 
            max_box, 
            sizeof(NMS_DT)*1, 
            NRAM2NRAM, 
            sizeof(NMS_DT)*save_str_num, 
            sizeof(NMS_DT)*1, 4);
    nram_save_count ++;
    output_box_num ++;
    
    if (dst != NRAM && output_box_num != 0) {
      store_dir = dst == SRAM ? NRAM2SRAM : NRAM2GDRAM;
      
      if (keep == keepNum) {
        __memcpy(output_data,
                nram_save,
                nram_save_count * sizeof(NMS_DT),
                store_dir,
                output_stride * sizeof(NMS_DT),
                nram_save_limit_count * sizeof(NMS_DT),
                4);
      }
    } 
    if (input_stride > input_box_num) {
      __nramset(inter_x1, seg_len, 0);
      int tail_len = input_stride - input_box_num;
      __memcpy(score + input_box_num,
              inter_x1,
              tail_len * sizeof(NMS_DT),
              NRAM2NRAM,
              tail_len * sizeof(NMS_DT),
              tail_len * sizeof(NMS_DT),
              0);
    }

    __nramset(inter_y1, seg_len, max_box[1]);
    __svmax_relu(inter_x1, input_x1_ptr, inter_y1, seg_len);
    __nramset(inter_y2, seg_len, max_box[3]);
    __svmin_relu(inter_x2, input_x2_ptr, inter_y2, seg_len);

    __bang_sub(inter_x1, inter_x2, inter_x1, seg_len);
    __bang_active_relu(inter_x1, inter_x1, seg_len);

    __nramset(inter_x2, seg_len, max_box[2]);
    __svmax_relu(inter_y1, input_y1_ptr, inter_x2, seg_len);
    __nramset(inter_x2, seg_len, max_box[4]);
    __svmin_relu(inter_y2, input_y2_ptr, inter_x2, seg_len);

    __bang_sub(inter_y1, inter_y2, inter_y1, seg_len);
    __bang_active_relu(inter_y1, inter_y1, seg_len);

    __bang_mul(inter_x1, inter_x1, inter_y1, seg_len);

    __bang_sub(inter_y1, input_x2_ptr, input_x1_ptr, seg_len);
    __bang_sub(inter_y2, input_y2_ptr, input_y1_ptr, seg_len);
    __bang_mul(inter_x2, inter_y1, inter_y2, seg_len);

    __nramset(inter_y1, seg_len, max_area);
    __bang_add(inter_x2, inter_x2, inter_y1, seg_len);
    __bang_sub(inter_x2, inter_x2, inter_x1, seg_len);

    __bang_mul_const(inter_x2, inter_x2, thresh_iou, seg_len);
    __bang_gt(inter_x1, inter_x2, inter_x1, seg_len);
    __bang_mul(score, score, inter_x1, seg_len);
  }
}

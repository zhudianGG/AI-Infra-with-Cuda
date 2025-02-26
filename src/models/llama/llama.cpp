#include "src/models/llama/llama.h"

template <typename T>
void Llama<T>::allocateCPUBuffer(int batch_size)
{
    h_input_ids_buf_ =
        allocator->Malloc(h_input_ids_buf_, sizeof(int) * 13, true);
    h_input_length_buf_ =
        allocator->Malloc(h_input_length_buf_, sizeof(int) * batch_size, true);
    h_history_length_buf_ = 
        allocator->Malloc(h_history_length_buf_, sizeof(int) * batch_size, true);
    h_context_length_buf_ = 
        allocator->Malloc(h_context_length_buf_, sizeof(int) * batch_size, true);
    h_sequence_lengths_ = 
        allocator->Malloc(h_sequence_lengths_, sizeof(int) * batch_size, true);
    h_finished_buf_ = allocator->Malloc(h_finished_buf_, sizeof(bool) * batch_size, true);
    for (int i = 0; i < batch_size; i++)
    {
        h_finished_buf_[i] = 0;
    }
    h_output_ids = allocator->Malloc(h_output_ids, sizeof(int) * batch_size, true);
}

// allocate gpu buffer
template <typename T>
void Llama<T>::allocateGPUBuffer(int batch_size)\
{
    step = new TensorWrapper<int>(CPU, getTensorType<int>(), {1});
    layer = new TensorWrapper<int>(CPU, getTensorType<int>(), {1}, &layer_id);
    // for context decoder
    context_decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 13, hidden_units});
    context_decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 13, hidden_units});
    // split from context_decoder_output
    context_decoder_lmhead_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 1, hidden_units});
    // for self decoder
    decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*batch size*/ 1, hidden_units});
    decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*batch size*/ 1, hidden_units});
    input_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {/*token num*/ 13});
    // for context decoder
    input_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    history_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    context_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    sequence_lengths = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    // kv cache buffer
    all_k_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    all_v_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    token_ids = new TensorWrapper<int>(GPU, getTensorType<T><int>(), {batch_size});
    is_finished = new TensorWrapper<bool>(GPU, getTensorType<bool>(), {batch_size});
    output_rmsnorm_weight = new TensorWrapper<T>(GPU, getTensorType<T>(), {hidden_units}, llama_weights->out_rmsnorm_weight.gamma);
    probs = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, vocab_size});
    unused_residual = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, hidden_units});
    // allocate buffer of above
    unused_residual->data = allocator->Malloc(unused_residual->data, sizeof(T) * 13 * hidden_units, false);
    

}
import tensorflow as tf
import numpy as np
import random

class Seq2Seq_Model():
    def __init__(self, rnn_size, num_layers, dim_video_feat, embedding_size, 
                    learning_rate, word_to_idx, mode, max_gradient_norm, 
                    use_attention, beam_search, beam_size, 
                    max_encoder_steps, max_decoder_steps):
        tf.set_random_seed(9487)
        np.random.seed(9487)
        random.seed(9487)

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dim_video_feat = dim_video_feat
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.word_to_idx = word_to_idx
        self.mode = mode
        self.max_gradient_norm = max_gradient_norm
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_encoder_steps = max_encoder_steps
        self.max_decoder_steps = max_decoder_steps

        self.vocab_size = len(self.word_to_idx)

        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder, seed=9487)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        tf.set_random_seed(9487)
        np.random.seed(9487)
        random.seed(9487)

        print ('Building model...')
        # ========== Define model's placeholder ==========
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        # ========== Define model's encoder ==========
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # Encoder embedding.
            encoder_inputs_flatten = tf.reshape(self.encoder_inputs, [-1, self.dim_video_feat])
            encoder_inputs_embedded = tf.layers.dense(encoder_inputs_flatten, self.embedding_size, use_bias=True)
            encoder_inputs_embedded = tf.reshape(encoder_inputs_embedded, [self.batch_size, self.max_encoder_steps, self.rnn_size])

            # Build RNN cell
            encoder_cell = self._create_rnn_cell()

            # Run Dynamic RNN
            #   encoder_outputs: [batch_size, max_time, num_units]
            #   encoder_state: [batch_size, num_units]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded, 
                sequence_length=self.encoder_inputs_length, 
                dtype=tf.float32)

        # ========== Define model's encoder ==========        
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length

            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("Using beamsearch decoding...")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            # A dense matrix to turn the top hidden states to logit vectors of dimension V.
            projection_layer = tf.layers.Dense(units=self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=9487))
            
            # Decoder embedding
            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, self.rnn_size], -0.1, 0.1, seed=9487), name='embedding_decoder')


            # Build RNN cell
            decoder_cell = self._create_rnn_cell()

            if self.use_attention:
                #定义要使用的attention机制。
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.rnn_size, 
                    memory=encoder_outputs, 
                    normalize=True,
                    memory_sequence_length=encoder_inputs_length)

                # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, 
                    attention_mechanism=attention_mechanism, 
                    attention_layer_size=self.rnn_size, 
                    name='Attention_Wrapper')

                #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
                decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=9487))

            # if self.mode == 'train':
            # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
            # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<bos>']), ending], 1)
            
            # Look up embedding:
            #   decoder_inputs: [batch_size, max_time]
            #   decoder_inputs_embedded: [batch_size, max_time, embedding_size]
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

            # Helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_inputs_embedded, 
                sequence_length=self.decoder_targets_length, 
                time_major=False, name='training_helper')
            # Decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper, 
                initial_state=decoder_initial_state, 
                output_layer=output_layer)
            
            # decoder_outputs: (rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]
            # sample_id: [batch_size], tf.int32
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, 
                impute_finished=True, 
                maximum_iterations=self.max_target_sequence_length)

            # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')

            # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.decoder_logits_train, 
                targets=self.decoder_targets, 
                weights=self.mask)

            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            # elif self.mode == 'decode':
            # Token of BOS and EOS.
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<bos>']
            end_token = self.word_to_idx['<eos>']
            
            # decoder阶段根据是否使用beam_search决定不同的组合，
            # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
            # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
            if self.beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell, 
                    embedding=embedding_decoder,
                    start_tokens=start_tokens, 
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_size,
                    output_layer=output_layer)
            else:
                # Helper
                inference_decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding_decoder, 
                    start_tokens=start_tokens, 
                    end_token=end_token)
                # Decoder
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, 
                    helper=inference_decoding_helper, 
                    initial_state=decoder_initial_state, 
                    output_layer=output_layer)

            # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
            # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]
            # sample_id: [batch_size, decoder_targets_length], tf.int32

            # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
            # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
            # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
            # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, 
                maximum_iterations=self.max_decoder_steps)

            if self.beam_search:
                self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
                self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output
            else:
                self.decoder_predict_decode = tf.expand_dims(inference_decoder_outputs.sample_id, -1)
                self.decoder_predict_logits = inference_decoder_outputs.rnn_output

        # ========== Define model's saver ==========
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 0.8,
                      self.batch_size: len(encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, encoder_inputs, encoder_inputs_length):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
        return predict, logits
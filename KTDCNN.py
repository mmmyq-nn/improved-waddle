# This script is used for multi-GPUs training and validation on ImageNet.

import os
import shutil
import imp
import time
import h5py
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from openpyxl import Workbook
import GetData_ImageNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# global params
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('flag_net_module', './ModuleResNeKt_V2.py', 'Module selection with specific dataset.')
flags.DEFINE_integer('flag_depth', 50, 'Depth of CNN, e.g., 50 implies ResNet-50 or some other similar CNN.')
flags.DEFINE_boolean('flag_asymmetric', False, 'Whether using asymmetric convolution in bottleneck.')
flags.DEFINE_boolean('flag_depthwise', False, 'Whether using depthwise convolution in bottleneck.')
flags.DEFINE_string('flag_log_dir', './log', 'Directory to put log files.')
flags.DEFINE_string('flag_dataset_dir', 'D:/Datasets/ImageNet/TFRecords', 'Directory of dataset.')
flags.DEFINE_integer('flag_max_epochs', 100, 'Maximum number of epochs to train.')
flags.DEFINE_integer('flag_batch_size', 25, 'Batch size of single GPU.')
flags.DEFINE_float('flag_learning_rate', 0.1, 'Learning rate to define the momentum optimizer.')
flags.DEFINE_float('flag_lr_decay', 30.0, 'Epochs to decay learning rate.')


# calculate average gradients from multi-GPUs
def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		grad_and_var = (grad, grad_and_vars[0][1])
		average_grads.append(grad_and_var)
	return average_grads


def run_training(b_gpu_enabled = False, str_restore_ckpt = None):
	network = imp.load_source('network', FLAGS.flag_net_module)

	with tf.Graph().as_default(), tf.device('/cpu:0'):
		# GPU amount
		n_num_gpus = 1
		if b_gpu_enabled == True:
			l_devices = device_lib.list_local_devices()
			for i in range(len(l_devices)):
				if l_devices[i].device_type == 'GPU':
					n_num_gpus += 1
			n_num_gpus -= 1

		# iteration step, initialize as 0
		tfv_global_step = tf.get_variable('var_global_step', [], tf.int32, tf.constant_initializer(0, tf.int32), trainable = False)

		# flag to indicate training (True) or validating (False)
		tfv_train_phase = tf.Variable(True, trainable = False, name = 'var_train_phase', dtype = tf.bool, collections = [])

		# EMA for all trainable variables
		tfob_variable_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg_variable')

		# momentum optimizer
		n_decay_steps = int(FLAGS.flag_lr_decay * GetData_ImageNet._NUM_IMAGES['train'] / FLAGS.flag_batch_size)
		f_learning_rate = tf.train.exponential_decay(FLAGS.flag_learning_rate, tfv_global_step, n_decay_steps, 0.1, staircase = True)
		optim = tf.train.MomentumOptimizer(f_learning_rate, 0.9)

		# getting data
		t_data, t_labels = GetData_ImageNet.input_fn(True, FLAGS.flag_dataset_dir, FLAGS.flag_batch_size * n_num_gpus, FLAGS.flag_max_epochs + 1)
		v_data, v_labels = GetData_ImageNet.input_fn(False, FLAGS.flag_dataset_dir, FLAGS.flag_batch_size * n_num_gpus, FLAGS.flag_max_epochs)
		t_data_split = tf.split(t_data, n_num_gpus)
		t_labels_split = tf.split(t_labels, n_num_gpus)
		v_data_split = tf.split(v_data, n_num_gpus)
		v_labels_split = tf.split(v_labels, n_num_gpus)

		# network inference
		tower_losses_t = []
		tower_evals_t = []
		tower_losses_v = []
		tower_evals_v = []
		tower_grads = []		
		for i in range(n_num_gpus):
			with tf.device('/gpu:%d' % i):
				loss_t, eval_t, loss_v, eval_v = network.get_network_output(
					i, t_data_split[i], t_labels_split[i], v_data_split[i], v_labels_split[i], 
					FLAGS.flag_depth, FLAGS.flag_asymmetric, FLAGS.flag_depthwise, tfv_train_phase)

				tower_losses_t.append(loss_t)
				tower_evals_t.append(eval_t)
				tower_losses_v.append(loss_v)
				tower_evals_v.append(eval_v)

				# using optimizer to calculate gradients
				grads = optim.compute_gradients(loss_t)

				# gradients clip
				grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if grad is not None]
				tower_grads.append(grads)

		# average gradients from multi-GPUs
		grads = average_gradients(tower_grads)

		# apply gradients by optimizer
		tfop_apply_gradients = optim.apply_gradients(grads, tfv_global_step)
		with tf.control_dependencies([tfop_apply_gradients]):
			# revise global steps caused by multi-GPUs
			tfop_normalize_gs = tfv_global_step.assign_add(n_num_gpus - 1)

		# apply EMA for all trainable variables
		tfop_variable_averages_apply = tfob_variable_averages.apply(tf.trainable_variables())

		# loss and evaluation in single epoch
		tfv_train_loss = tf.Variable(5.0, trainable = False, name = 'var_train_loss', dtype = tf.float32)
		tfv_train_precision = tf.Variable(0.0, trainable = False, name = 'var_train_precision', dtype = tf.float32)

		# updating for tfv_train_loss and tfv_train_precision
		l_ops_train_lp_update = []
		for i in range(n_num_gpus):
			l_ops_train_lp_update.append(tfv_train_loss.assign_sub(0.1 * (tfv_train_loss - tower_losses_t[i])))
			new_precision = tf.reduce_mean(tf.cast(tower_evals_t[i], tf.float32))
			l_ops_train_lp_update.append(tfv_train_precision.assign_sub(0.1 * (tfv_train_precision - new_precision)))
		tfop_train_lp_update = tf.group(*l_ops_train_lp_update)

		# group all the above operations
		tfop_train = tf.group(tfop_apply_gradients, tfop_normalize_gs, tfop_variable_averages_apply, tfop_train_lp_update)

		# model saver
		tfob_saver = tf.train.Saver(tf.global_variables())
		tfob_saver_ema = tf.train.Saver(tfob_variable_averages.variables_to_restore())

		# Session
		if b_gpu_enabled == True:
			tfob_sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True, per_process_gpu_memory_fraction = 0.99)))
		else:
			tfob_sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, device_count = {'GPU': 0}))

		# run initialization of all variables
		tfob_sess.run(tf.global_variables_initializer())

		# number of the beginning epoch and the amount of steps in one epoch
		n_epoch_steps = int(GetData_ImageNet._NUM_IMAGES['train'] / FLAGS.flag_batch_size + 0.5)
		n_start_epoch = 0
		if str_restore_ckpt is not None:
			tfob_saver.restore(tfob_sess, str_restore_ckpt)
			print('Previously started training session restored from "%s".' % str_restore_ckpt)
			n_start_epoch = int(tfob_sess.run(tfv_global_step)) // n_epoch_steps
		print('Starting with epoch #%d.\n' % (n_start_epoch + 1))

		# loss and val recorded
		l_rc_loss_pre = []
		if os.path.exists(FLAGS.flag_log_dir + '/learning_curve.h5'):
			with h5py.File(FLAGS.flag_log_dir + '/learning_curve.h5', 'r') as file:
				arr_rc_loss_pre = file.get('curve').value
			l_rc_loss_pre = arr_rc_loss_pre.tolist()

		# begin the training loop, from n_start_epoch to flag_max_epochs
		for n_epoch in range(n_start_epoch, FLAGS.flag_max_epochs):
			cur_loss_pre = []

			# -------------------------------------------------------------------------------------------------
			# Training begin! set training flag to True.
			tfob_sess.run(tfv_train_phase.assign(True))
			print('Epoch #%d. [Train]' % (n_epoch + 1))
			
			# step in single epoch
			n_step = 0

			# training process of current epoch
			while n_step < n_epoch_steps:
				# run training
				_, loss_train, eval_train = tfob_sess.run([tfop_train, loss_t, eval_t])
				assert not np.isnan(loss_train), 'Model diverged with loss = NaN.'				
				n_step += n_num_gpus
				print('Epoch #%d [Train]. Step %d/%d. Batch loss = %.2f. Batch precision = %.2f.' % (n_epoch + 1, n_step, n_epoch_steps, loss_train, np.mean(eval_train) * 100.0))
				
			# Training end! evaluate current result and record checkpoint
			train_loss_value, train_precision_value = tfob_sess.run([tfv_train_loss, tfv_train_precision])
			print('Epoch #%d. Train loss = %.2f. Train precision = %.2f.' % (n_epoch + 1, train_loss_value, train_precision_value * 100.0))
			cur_loss_pre += [train_loss_value, train_precision_value * 100.0]
			str_checkpoint_path = os.path.join(FLAGS.flag_log_dir, 'model.ckpt')
			str_ckpt = tfob_saver.save(tfob_sess, str_checkpoint_path, tfv_global_step)
			print('Checkpoint "%s" is saved.\n' % str_ckpt)
			# -------------------------------------------------------------------------------------------------

			# -------------------------------------------------------------------------------------------------
			# Evaluate begin! set training flag to False, and use EMA to restore model
			tfob_sess.run(tfv_train_phase.assign(False))
			print('Epoch #%d. [Evaluation]' % (n_epoch + 1))
			tfob_saver_ema.restore(tfob_sess, str_ckpt)
			print('EMA variables restored.')

			# capacity of validation set and calculate steps according to it
			n_val_count = GetData_ImageNet._NUM_IMAGES['validation']
			n_val_steps = (n_val_count + FLAGS.flag_batch_size - 1) // FLAGS.flag_batch_size

			# positive examples and losses
			n_val_corrects = 0
			n_val_losses = 0.0

			# validating process of current epoch
			while n_val_count > 0:
				# run validation
				eval_validation_and_loss_validation = tfob_sess.run(tower_evals_v + tower_losses_v)
				eval_validation = np.concatenate(eval_validation_and_loss_validation[:n_num_gpus], axis = 0)
				loss_validation = eval_validation_and_loss_validation[-n_num_gpus:]
				n_cnt = min(eval_validation.shape[0], n_val_count)
				n_val_count -= n_cnt
				n_cur_step = n_val_steps - (n_val_count + FLAGS.flag_batch_size - 1) // FLAGS.flag_batch_size

				# accumulate positive examples and losses
				n_val_corrects += np.sum(eval_validation[:n_cnt])
				n_val_losses += np.sum(loss_validation) * FLAGS.flag_batch_size
				print('Epoch #%d [Evaluation]. Step %d/%d. Batch loss = %.2f. Batch precision = %.2f.' % (n_epoch + 1, n_cur_step, n_val_steps, np.mean(loss_validation), np.mean(eval_validation) * 100.0))

			# Evaluate end! evaluate current result and restore checkpoint without EMA for the next training
			validation_precision_value = n_val_corrects / GetData_ImageNet._NUM_IMAGES['validation']
			validation_loss_value = n_val_losses / GetData_ImageNet._NUM_IMAGES['validation']
			print('Epoch #%d. Validation loss = %.2f. Validation precision = %.2f.' % (n_epoch + 1, validation_loss_value, validation_precision_value * 100.0))
			cur_loss_pre += [validation_loss_value, validation_precision_value * 100.0]
			tfob_saver.restore(tfob_sess, str_ckpt)
			print('Variables restored.\n')
			# -------------------------------------------------------------------------------------------------

			# record the loss and val of current epoch
			l_rc_loss_pre.append(cur_loss_pre)
			with h5py.File(FLAGS.flag_log_dir + '/learning_curve.h5', 'w') as file:
				file.create_dataset('curve', data = np.array(l_rc_loss_pre, dtype = np.float32))

		# record the final loss and precision
		wb = Workbook()
		ws = wb.create_sheet()
		for line in l_rc_loss_pre:
			ws.append(line)
		wb.save('learning_curve.xlsx')
		wb.close()

	return None


def main(_):
	b_gpu_enabled = False
	l_devices = device_lib.list_local_devices()
	for i in range(len(l_devices)):
		if l_devices[i].device_type == 'GPU':
			if l_devices[i].memory_limit > 2 * 1024 * 1024 * 1024 :
				b_gpu_enabled = True
				break

	str_last_ckpt = tf.train.latest_checkpoint(FLAGS.flag_log_dir)
	if str_last_ckpt is not None:
		while True:
			print('Checkpoint "%s" found. Continue last training session?' % str_last_ckpt)
			print('Continue - [c/C]. Restart (all content of log dir will be removed) - [r/R]. Abort - [a/A].')
			ans = input().lower()
			if len(ans) == 0:
				continue
			if ans[0] == 'c':
				break
			elif ans[0] == 'r':
				str_last_ckpt = None
				shutil.rmtree(FLAGS.flag_log_dir)
				time.sleep(1)
				break
			elif ans[0] == 'a':
				return

	if os.path.exists(FLAGS.flag_log_dir) == False:
		os.mkdir(FLAGS.flag_log_dir)

	run_training(b_gpu_enabled, str_last_ckpt)
	print('Program is finished.')


if __name__ == '__main__':
    tf.app.run()

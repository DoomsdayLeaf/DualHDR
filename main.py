from __future__ import print_function
import argparse, os
import tensorflow as tf

from DataProcess import convert_to_tf_record, convert_eye_data_to_list_dir, convert_real_data_to_list_dir, convert_exr_to_hdr, convert_format
from SingleImageNet import train_dequantization_net, train_linearization_net, train_hallucination_net
from SingleImageNet.Dual_Deq import train_dual_dequantization
from SingleImageNet.Joint_Deq_Lin import train_joint_deq_lin
from SingleImageNet.Dual_Hal import train_dual_hal
from SingleImageNet.Dual_Ref import train_dual_refinement, test_now


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # change ID to designate GPU
#gpu_device = '/gpu:0,1,2,3'

"""parsing and configuration"""
def parse_args():
    desc = "Official Tensorflow Implementation of HDR CNN"
    parser = argparse.ArgumentParser(description=desc)

    """ Training Settings """
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'run_data_convert'])
    parser.add_argument('--train_mode', type=str, default='train_OONet',
                        choices=['train_Dnet', 'train_INet',
                                 'train_ITnet', 'train_OONet',])
    parser.add_argument('--train_data_path_SDR', type=str, default='../train_dataset/HDR-Real/LDR_in/', help='Train input data path')
    parser.add_argument('--train_data_path_HDR', type=str, default='../train_dataset/HDR-Real/HDR_gt/', help='Train GT data path')


    """ base config """
    parser.add_argument('--hdr_prefix', type=str, default='data/HDR-Synth/', help='Single Image HDR train data prefix')
    parser.add_argument('--coverHDR_path', type=str, default='data/HDR-Real/', help='Single Image HDR Real data')
    parser.add_argument('--test_real_data', type=str, default='result/HDR-Real/LDR/',help='Single Image HDR HDR-Real test path')
    parser.add_argument('--test_eye_data', type=str, default='result/HDR-Eye/LDR/',help='Single Image HDR HDR-Eye test path')

    """ Dnet """
    parser.add_argument('--deq_batch_size', type=int, default=8, help='The size of Dnet batch size.')
    parser.add_argument('--deq_epoch', type=int, default=500000, help='The number of Dnet epochs to run')
    parser.add_argument('--deq_indi_log_dir', type=str, default='ckpt/ckpt_indi_dnet/', help='Dnet train model result')
    parser.add_argument('--deq_log_dir', type=str, default='ckpt/ckpt_dnet/', help='Dnet train model result')
    parser.add_argument('--deq_joint_train', type=str, default="indi",
                        choices=["indi", "joint_pair", "joint_unpair"], help='Dual train mode')

    """ Dual Non-linear Mapping """
    parser.add_argument('--linear_batch_size', type=int, default=8, help='The size of linear net batch size.')
    parser.add_argument('--linear_epoch', type=int, default=100000, help='The number of linear net epochs to run')
    parser.add_argument('--linear_indi_log_dir', type=str,
                        default='ckpt/ckpt_indi_inet/', help='linear net train model result')
    parser.add_argument('--linear_log_dir', type=str,
                        default='ckpt/ckpt_inet/', help='linear net train model result')
    parser.add_argument('--non_linear_log_dir', type=str,
                        default='ckpt/ckpt_nnet/', help='non-linear net train model result')
    parser.add_argument('--inet_joint_train', type=str, default="indi",
                        choices=["indi", "joint_pair", "joint_unpair"], help='Dual train mode')

    """ ITnet """
    parser.add_argument('--hal_batch_size', type=int, default=8, help='The size of ITnet batch size.')
    parser.add_argument('--hal_epoch', type=int, default=500000, help='The number of ITnet epochs to run')
    parser.add_argument('--hal_indi_log_dir', type=str, default='ckpt/ckpt_indi_itnet/', help='ITnet train model result')
    parser.add_argument('--hal_log_dir', type=str, default='ckpt/ckpt_itnet/', help='ITnet train model result')
    parser.add_argument('--itnet_joint_train', type=str, default="indi",
                        choices=["indi", "joint_pair", "joint_unpair"], help='Dual train mode')

    """ finetune real net """
    parser.add_argument('--ref_batch_size', type=int, default=8, help='The size of oo net batch size.')
    parser.add_argument('--ref_epoch', type=int, default=10000, help='The number of oo net epochs to run')
    parser.add_argument('--ref_indi_log_dir', type=str, default='ckpt/ckpt_indi_oonet/', help='oo net train model result')
    parser.add_argument('--ref_log_dir', type=str, default='ckpt/ckpt_oonet/', help='oo net train model result')
    parser.add_argument('--oonet_joint_train', type=str, default="indi",
                        choices=["indi", "joint_pair", "joint_unpair"], help='Dual train mode')


    parser.add_argument('--tf_records_log_dir', type=str,
                        default='tf_records/256_64_b32_tfrecords/', help='hdr real data set path')

    """ test """
    parser.add_argument('--test_dataset', type=str, default='dataset/HDR-Eye/', help='test dataset')
    parser.add_argument('--test_output_path', type=str, default='result/HDR-Eye/', help='output for testing')
    parser.add_argument('--pre_dnet', type=str, default='ckpt/ckpt_dnet/', help='Dnet model')
    parser.add_argument('--pre_inet', type=str, default='ckpt/ckpt_inet/', help='INet model')
    parser.add_argument('--pre_itnet', type=str, default='ckpt/ckpt_itnet/', help='ITnet model')
    parser.add_argument('--pre_oonet', type=str, default='ckpt/ckpt_oonet/', help='oonet model')

    return parser.parse_args()

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
    
def show_all_variables():
    model_vars = tf.trainable_variables()
    #slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    return model_vars


def main():
    args = parse_args()
    if args is None:
        exit()

    if args.phase == 'train':
        """ Open session """
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            train_mode = args.train_mode
            print("[*] training starts:  Starting training %s"%(train_mode))
            if train_mode == 'train_Dnet':
                if args.deq_joint_train == 'indi':
                    train_net = train_dequantization_net.train_dequantization_net(sess, args)
                else:
                    train_net = train_dual_dequantization.train_dual_deq_net(sess, args)
                train_net.train()
            elif train_mode == 'train_INet':
                if args.deq_joint_train == 'indi':
                    train_net = train_linearization_net.train_linearization_net(sess, args)
                else:
                    train_net = train_joint_deq_lin.Joint_Deq_Lin_Model(sess, args)
                train_net.train()
            elif train_mode == 'train_ITnet':
                if args.deq_joint_train == 'indi':
                    train_net = train_hallucination_net.train_hallucination_net(sess, args)
                else:
                    train_net = train_dual_hal.Dual_Hal_Real(sess, args)
                train_net.train()
            elif train_mode == 'train_OONet':
                train_net = train_dual_refinement.Finetune_real_dataset(sess, args)
                train_net.train()
            else:
                print("no mode to train")
            print("[*] training finished! ")

    elif args.phase == 'test':
        """ Open session """
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            print("Start testing HDR on %s" % (args.test_dataset))
            test_now_model = test_now.Test_Eye(sess, args, config)
            test_now_model.run_test()
            print("finish test")

    elif args.phase == 'run_data_convert':
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            convert_mode = 'convert_exr_to_hdr'
            dataset = 'RAISE'

            print("start converting")

            if convert_mode == 'tf_records':
                print("Start tf_records")
                conv_model = convert_to_tf_record.convert_to_tf_record(sess, args)
                conv_model.run_convert()
                print("Finish tf_records")

            elif convert_mode == 'data_list':
                print("Start %s" % (dataset))
                if dataset == 'RAISE':
                    convert_model = convert_real_data_to_list_dir.Convert_To_List_Dir(sess, args)
                elif dataset == 'eye':
                    convert_model = convert_eye_data_to_list_dir.Convert_To(sess, args)
                convert_model.run_convert()
                print("Finish %s" % (dataset))

            elif convert_mode == 'convert_exr_to_hdr':
                print("Start convert_exr_to_hdr")
                convert_model = convert_exr_to_hdr.Convert_To(sess, args)
                convert_model.run_convert()
                print("Finish convert_exr_to_hdr")

            elif convert_mode == 'format':
                print("Start format")
                convert_model = convert_format.Convert_To(sess, args)
                convert_model.run_convert()
                print("Finish format")

            print("finish converting")



if __name__ == '__main__':
    main()

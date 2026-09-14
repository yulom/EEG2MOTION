import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.5, type=float, help="learning rate decay")
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=8192, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=51, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--num-local-layer", type=int, default=2, help="nb of transformer local layers")
    parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='VQVAE', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    
    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')

    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay') #e-6
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')

    ## eeg_setting
    parser.add_argument('--t2m_train_data_path', type=str, default='EEG_data/HumanML3D_train_processed.pt')
    parser.add_argument('--t2m_test_data_path', type=str, default='EEG_data/HumanML3D_test_processed.pt')
    parser.add_argument('--t2m_val_data_path', type=str, default='EEG_data/HumanML3D_val_processed.pt')
    parser.add_argument('--video_feat_root', type=str, default='EEG_data/videomae_global_features')
    parser.add_argument('--motion_generation_method', type=str, default='from_scratch', choices=['finetuning','from_scratch'])
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
    parser.add_argument('--batch-size', default=512, type=int, help='batch size')#finetuning 256,from_scratch512
    parser.add_argument('--clip_loss_type', type=str, default='cosine') #dist,cosine
    parser.add_argument('--clip_target', type=str, default='video_mae') #motion_t2m,text_clip,video_mae
    parser.add_argument('--clip_loss', type=bool, default=True)
    parser.add_argument('--test_avg', type=bool, default=False)
    parser.add_argument('--clip_weight',type=float, default=1.)
    parser.add_argument('--pose_clip_weight',type=float, default=0.)
    parser.add_argument('--teacher', type=float, default=0.)
    parser.add_argument('--clip_temp',type=float, default=0.07)
    parser.add_argument('--teacher_temp',type=float, default=3.)
    parser.add_argument('--eeg_ch', type=str, default='all') #all,only_motor,only_visual,only_frontal,wo_motor,wo_visual,wo_frontal
    parser.add_argument('--eeg_data_root', type=str, nargs='+', default=['EEG_data/20260416S1','EEG_data/20260420S2', 'EEG_data/20260430S4','EEG_data/20260511S6','EEG_data/20260512S7','EEG_data/20260515S9'])
    # parser.add_argument('--eeg_data_root', type=str, nargs='+', default=['EEG_data/20260416S1','EEG_data/20260420S2', 'EEG_data/20260423S3', 'EEG_data/20260430S4','EEG_data/20260508S5','EEG_data/20260511S6','EEG_data/20260512S7','EEG_data/20260513S8','EEG_data/20260515S9'])
    parser.add_argument('--eeg_data_name', type=str, default='_final.mat')
    parser.add_argument("--motion_pool_size", type=int, default=100)
    parser.add_argument('--cls_head', type=bool, default=True)
    # parser.add_argument('--CFG', type=bool, default=True)
    # parser.add_argument('--CFG_prob',type=float, default=0.1)
    # parser.add_argument('--CFG_weight',type=float, default=2.0)
    parser.add_argument("--val_per_epoch", type=int, default=50)
    parser.add_argument('--pkeep', type=float, default=0.5, help='keep rate for gpt training')
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--random_level', type=bool, default=False)
    parser.add_argument('--eeg_embed_dim', default=256, type=int)
    parser.add_argument('--eeg_num_local_layer', default=2, type=int)
    parser.add_argument('--eeg_drop_out_rate', type=float, default=0.1)
    parser.add_argument("--eeg_ff_rate", type=int, default=4)
    parser.add_argument("--eeg_num_layers", type=int, default=2)
    parser.add_argument("--eeg_n_head", type=int, default=4)
    parser.add_argument("--eeg_iter", type=int, default=10000)
    parser.add_argument("--eeg_milestones", default=[8000], nargs="+", type=int)




    return parser.parse_args()
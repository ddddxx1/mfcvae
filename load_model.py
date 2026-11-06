
from mfcvae import MFCVAE
import torch
from torchvision.datasets import MNIST, SVHN
from datasets import Fast_3DShapes
from datasets_enrc import Fast_CMNIST, Fast_Stickfigures, Fast_GTSRB, Fast_NRObjects
import torchvision
from torchvision import transforms


def load_model_from_save_dict(save_dict_path, map_location='cuda:0'):

    try:
        save_dict = torch.load(save_dict_path, map_location=map_location, weights_only=False)
    except TypeError:
        save_dict = torch.load(save_dict_path, map_location=map_location)
    state_dict = save_dict['state_dict']
    args = save_dict['args']
    args.device = map_location

    device = torch.device(args.device)

     # dataset-specific parameters
    if args.dataset in ['fast_mnist']:
        height, width = 28, 28
        n_true_classes = 10
        cmap = 'grey'
        in_channels = 1
        in_dim = 28 * 28 * in_channels
        likelihood_model = 'Bernoulli'
        activation_x_hat_z = "sigmoid"
        n_labels = 1
    elif args.dataset in ['fast_svhn']:
        height, width = 32, 32
        n_true_classes = 10
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = "sigmoid"    # BE CAREFUL!!!
        n_labels = 1
    elif args.dataset == 'fast_3dshapes':
        n_labels = len(args.factors_label_list)
        chosen_attr_list = args.factors_label_list
        n_true_classes = []
        for j in range(n_labels):
            n_true_classes.append(len(args.factors_variation_dict[args.factors_label_list[j]]))
            assert len(args.factors_variation_dict[args.factors_label_list[j]]) == n_true_classes[j]
        # print('number of true classes across labels: ', n_true_classes)
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = None  # 'sigmoid'  # 'sigmoid'  #  None  # 'sigmoid'
        n_labels = len(args.factors_label_list)
    elif args.dataset == 'fast_cmnist':
        target_size = getattr(args, 'cmnist_image_size', 56)
        height, width = target_size, target_size
        n_labels = 2
        chosen_attr_list = ['left_digit', 'right_digit']
        n_true_classes = [10, 10]
        cmap = 'grey'
        in_channels = 1
        in_dim = height * width * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = 'sigmoid'
    elif args.dataset == 'fast_gtsrb':
        height, width = 32, 32
        n_labels = 2
        chosen_attr_list = ['type', 'color']
        n_true_classes = [4, 2]
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = 'sigmoid'
    elif args.dataset == 'fast_nr_objects':
        height, width = 32, 32
        n_labels = len(args.factors_label_list)
        chosen_attr_list = list(args.factors_label_list)
        n_true_classes = [len(Fast_NRObjects.LABEL_NAMES[Fast_NRObjects.FACTOR_NAMES.index(name)])
                          for name in chosen_attr_list]
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = 'sigmoid'
    elif args.dataset == 'fast_stickfigures':
        height, width = 20, 20
        n_labels = 3
        chosen_attr_list = ['posture_0', 'posture_1', 'posture_2']
        n_true_classes = [3, 3, 3]
        cmap = 'grey'
        in_channels = 1
        in_dim = 20 * 20 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = 'sigmoid'

    # other dataset-specific hyperparams
    if args.dataset == 'fast_mnist':
        train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_svhn':
        train_data = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor())
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_3dshapes':
        train_data = Fast_3DShapes(train=True, device=args.device, train_frac = args.threedshapes_train_frac,
                                                     factors_variation_dict=args.factors_variation_dict,
                                                     factors_label_list=args.factors_label_list)
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_cmnist':
        target_size = getattr(args, 'cmnist_image_size', 56)
        train_data = Fast_CMNIST(train=True, target_size=target_size,
                                 train_frac=getattr(args, 'cmnist_train_frac', 5/6),
                                 seed=getattr(args, 'seed', 42), device='cpu')
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_gtsrb':
        train_data = Fast_GTSRB(root=getattr(args, 'gtsrb_root', 'data/enrc_data/gtsrb'),
                                train=True, train_frac=getattr(args, 'gtsrb_train_frac', 0.8),
                                seed=getattr(args, 'seed', 42), device='cpu')
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_nr_objects':
        train_data = Fast_NRObjects(root=getattr(args, 'nr_objects_root', 'data/enrc_data/nr_objects'),
                                train=True, train_frac=getattr(args, 'nr_objects_train_frac', 0.8),
                                seed=getattr(args, 'seed', 42), device='cpu')
        n_train_batches_per_epoch = (len(train_data) + args.batch_size - 1) // args.batch_size
    elif args.dataset == 'fast_stickfigures':
        train_data = Fast_Stickfigures(train=True, train_frac=getattr(args, 'stickfigures_train_frac', 0.8),
                                       seed=getattr(args, 'seed', 42), device='cpu')
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    # print("n_train_batches_per_epoch: ", n_train_batches_per_epoch)


    mfcvae = MFCVAE(in_dim=in_dim, J_n_mixtures=args.J_n_mixtures, z_j_dim_list=args.z_j_dim_list,
                    n_clusters_j_list=args.n_clusters_j_list,
                    device=device, encode_layer_dims=args.encode_layer_dims,
                    decode_layer_dims=args.decode_layer_dims,
                    activation_x_hat_z=activation_x_hat_z, likelihood_model=likelihood_model,
                    sigma_multiplier_p_x_z=args.sigma_multiplier_p_x_z,
                    cov_type_p_z_c=args.cov_type_p_z_c,
                    init_type_p_z_c=args.init_type_p_z_c,
                    init_off_diag_cov_p_z_c=args.init_off_diag_cov_p_z_c,
                    model_type=args.model_type,
                    fix_pi_p_c=args.fix_pi_p_c,
                    facet_to_fix_pi_p_c=args.facet_to_fix_pi_p_c if 'facet_to_fix_pi_p_c' in (vars(args)).keys() else 'all',  # does nothing if 'fix_pi_p_c' is False
                    n_train_batches_per_epoch=n_train_batches_per_epoch,
                    n_epochs_per_progressive_step=args.n_epochs_per_progressive_step, n_batches_fade_in=args.n_batches_fade_in, gamma_kl_z_pretrain=args.gamma_kl_z_pretrain, gamma_kl_c_pretrain=args.gamma_kl_c_pretrain,
                    do_progressive_training=args.do_progressive_training,
                    activation=args.activation if 'activation' in (vars(args)).keys() else 'relu',
                    do_fc_batch_norm=args.do_fc_batch_norm)

    mfcvae.load_state_dict(state_dict)

    return mfcvae, args

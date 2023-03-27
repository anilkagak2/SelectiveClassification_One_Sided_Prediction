
import os, sys, time, torch, random, argparse, json, math, shutil, copy, PIL
import itertools
from collections import namedtuple, defaultdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

from model_dict import get_model_from_name, get_model_infos
from osp_model import OSPModel, osp_loss
from get_dataset_with_transform import get_datasets
from log_utils import Logger, AverageMeter, ProgressMeter

def m__get_prefix( args ):
    prefix = '-'.join( [ 'OSP', args.dataset, args.method, args.model_name, str(args.epochs), str(args.mu), '--' ] )
    return prefix

def get_model_prefix( args ):
    prefix = './models/' + m__get_prefix( args ) 
    return prefix

def get_mlr(lr_scheduler):
     return lr_scheduler.optimizer.param_groups[0]['lr']

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+'model_best.pth.tar')

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def prepare_logger(xargs):
    args = copy.deepcopy(xargs)

    logger = Logger(args.save_dir, args.rand_seed)
    logger.log("Main Function with logger : {:}".format(logger))
    logger.log("Arguments : -------------------------------")
    for name, value in args._get_kwargs():
        logger.log("{:16} : {:}".format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def execute_network( network, inputs, train=False ):
    if train:
      logits, aux_logits = network(inputs)
    else:
      with torch.no_grad():
        logits, aux_logits = network(inputs)

    model_logits = {}
    model_logits['logits'] = logits
    model_logits['aux_logits'] = aux_logits
    return model_logits

def get_model_predictions( args, logger, model, xloader, batch_size, n_examples ):
    y_scores = np.zeros( (n_examples, args.class_num), dtype=np.float32 )
    y_correct = np.zeros( (n_examples,) )

    model.eval()
    for i, (inputs, targets) in enumerate(xloader):
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        bstart = batch_size * i
        bend = min(bstart + batch_size, n_examples)

        model_logits = execute_network( model, inputs, train=False )
        logits = model_logits['logits']
        y_out = F.softmax( logits, dim=1 )

        y_correct[ bstart:bend ] = ( torch.argmax( y_out, dim=1 ) == targets ).detach().cpu().numpy() 
        y_scores[ bstart:bend ] = y_out.detach().cpu().numpy() 
    return y_scores, y_correct

def gather_all_predictions( args, logger, xloader, batch_size, n_examples ):
    predictions = {}
    for mu in args.mus_list:
        args.mu = mu
        model_path = get_model_prefix( args ) + 'model_best.pth.tar'
        model = get_model( args ) 

        state = torch.load(model_path)
        model.load_state_dict( state['state_dict'] )

        predictions[mu] = get_model_predictions( args, logger, model, xloader, batch_size, n_examples )
        del model

    return predictions

def get_coverage_error_for_given_threshold( predictions, mu, th ):
    y_scores, y_correct = predictions[mu]
    belongs_to_class = np.sum( y_scores >= th, axis=1 )

    n_examples = len( y_correct )
    coverage = (1. * np.sum( belongs_to_class > 0 )) / n_examples 
    accuracy = (1. * np.sum( (belongs_to_class > 0) * y_correct )) / n_examples
    error = coverage - accuracy
    return error, coverage

def update_best_params_for_cov( args, logger, D, cur_coverage, cur_error, mu, th ):
    for cov in args.coverage_list:
        best_cov, best_error, best_mu, best_th = D[ cov ]
        if (cur_coverage >= cov) and (cur_error < best_error):
            D[ cov ] = (cur_coverage, cur_error, mu, th)  

def update_best_params_for_error( args, logger, D, cur_coverage, cur_error, mu, th ):
    for error in args.error_list:
        best_cov, best_error, best_mu, best_th = D[ error ]
        if (cur_error <= error) and (cur_coverage > best_cov):
            D[ error ] = (cur_coverage, cur_error, mu, th)  

def search_for_target_cov_error( args, logger, predictions ):
    thresholds = np.linspace(0, 1, num=args.n_threshold)

    logger.log("-" * 50)
    logger.log( 'target errors = ' + str( args.error_list ) )
    logger.log( 'target coverage = ' + str( args.coverage_list ) )
    logger.log( 'mus = ' + str(args.mus_list) )
    logger.log("-" * 50)

    best_cov_at_error_dict = defaultdict( lambda: (0.0, 0.0, None, None) )
    best_error_at_cov_dict = defaultdict( lambda: (0.0, 100.0, None, None) )

    for mu in args.mus_list:
        for th in thresholds:
            cur_error, cur_coverage = get_coverage_error_for_given_threshold( predictions, mu, th )

            update_best_params_for_cov( args, logger, best_error_at_cov_dict, cur_coverage, cur_error, mu, th )       
            update_best_params_for_error( args, logger, best_cov_at_error_dict, cur_coverage, cur_error, mu, th )

    for error in args.error_list:
        best_cov, best_error, best_mu, best_th = best_cov_at_error_dict[ error ]
        if best_mu is not None:
            logger.log('For desired_error={:.4}, ==> test cov={:.4}, err=={:.4} with mu={:.4}, th={:.4}'.format(error, best_cov, best_error, 
                                    best_mu, best_th) )
        else:
            logger.log('For desired_error={:.4}, could not find any parameters'.format(error))
    logger.log("-" * 50)

    for cov in args.coverage_list:
        best_cov, best_error, best_mu, best_th = best_error_at_cov_dict[ cov ]
        if best_mu is not None:
            logger.log('For desired_coverage={:.4}, ==> test cov={:.4}, err=={:.4} with mu={:.4}, th={:.4}'.format(cov, best_cov, best_error, 
                                    best_mu, best_th) )
        else:
            logger.log('For desired_coverage={:.4}, could not find any parameters'.format(cov))
    logger.log("-" * 50)

def evaluate_coverage_error( args, logger, xloader, batch_size ):
    n_examples = len( xloader.dataset )
    predictions = gather_all_predictions( args, logger, xloader, batch_size, n_examples )
    search_for_target_cov_error( args, logger, predictions )

def train_eval_loop( args, logger, epoch, model, optimizer, scheduler, max_optimizer, max_scheduler, xloader, criterion, batch_size, mode='eval' ):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress_bar_list = [ losses, top1, top5, ]

    if mode == 'eval': 
        model.eval()
    else:
        model.train()

    progress = ProgressMeter(
            logger,
            len(xloader),
            progress_bar_list, 
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets) in enumerate(xloader):
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        if mode == 'eval':
            model_logits = execute_network( model, inputs, train=False )
            logits = model_logits['logits']
            # criterion( logits, targets ) 
            loss = osp_loss( args, model, model_logits, targets )
        else:
            optimizer.zero_grad()
            max_optimizer.zero_grad()

            model_logits = execute_network( model, inputs, train=True )
            logits = model_logits['logits']

            # criterion( logits, targets ) 
            loss = osp_loss( args, model, model_logits, targets, mu=args.mu )

            loss.backward()

            # Change gradient sign for the max step
            model.lambdas.grad *= -1.

            optimizer.step()
            max_optimizer.step()

            if (i == len(xloader)-1):
                scheduler.step()
                max_scheduler.step()

        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg

def get_optimizer_scheduler( parameters, args, lr ):
    optimizer = torch.optim.AdamW(parameters, lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return optimizer, scheduler

def get_model(args):
    model = get_model_from_name( args.class_num, args.model_name )
    model = OSPModel( model, num_classes=args.class_num, mu=args.mu )
    model = model.cuda()
    return model

def get_model_optimizer_and_scheduler( args, logger, xshape ):
    model = get_model( args )
    flop, param = get_model_infos(model, xshape)
    del model

    model = get_model( args )
    optimizer, scheduler = get_optimizer_scheduler( model.get_minimization_vars(), args, args.lr )
    max_optimizer, max_scheduler = get_optimizer_scheduler( model.get_maximization_vars(), args, args.max_lr )
    return model, optimizer, scheduler, max_optimizer, max_scheduler, flop, param 

def train_one_model_for_mu( args, logger, xshape, mu, train_loader, valid_loader ):
    args.mu = mu
    args.save_dir = args.save_dir + m__get_prefix( args ) 
    criterion = nn.CrossEntropyLoss()

    model, optimizer, scheduler, max_optimizer, max_scheduler, flop, param = get_model_optimizer_and_scheduler(args, logger, xshape)

    if args._ckpt != "" and len(args._ckpt)>3:
        state = torch.load(args._ckpt)
        model.load_state_dict( state['state_dict'] )

    logger.log("-" * 50)
    logger.log("mu = {:}".format( mu ))
    logger.log("model information : {:}".format(model.get_message()))
    logger.log("-" * 50)
    logger.log("Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format( param, flop, flop / 1e3 ) )

    epoch=-1
    val_loss, val_acc1,_ = train_eval_loop( args, logger, epoch, model, None, None, None, None, valid_loader, criterion, args.eval_batch_size, mode='eval' )
    best_acc = val_acc1
    best_state_dict = model.state_dict()
    logger.log(' -- Best acc so far ' + str( best_acc ))

    for epoch in range(args.epochs):
        trn_loss, trn_acc1,_ = train_eval_loop( args, logger, epoch, model, optimizer, scheduler, max_optimizer, max_scheduler, train_loader, criterion, args.batch_size, mode='train')
        val_loss, val_acc1,_ = train_eval_loop( args, logger, epoch, model, None, None, None, None, valid_loader, criterion, args.eval_batch_size, mode='eval')

        is_best = False
        if val_acc1 > best_acc:
            best_acc = val_acc1
            is_best = True
            best_state_dict = model.state_dict()

        state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'scheduler' : scheduler.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'max_scheduler' : max_scheduler.state_dict(),
                'max_optimizer' : max_optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, prefix=get_model_prefix( args ) )
        logger.log( '\t\t LR=' + str(get_mlr(scheduler)) + ' -- best acc so far ' + str( best_acc ) )

    model.load_state_dict( best_state_dict ) 

    val_loss, val_acc1,_ = train_eval_loop( args, logger, epoch, model, None, None, None, None, valid_loader, criterion, args.eval_batch_size, mode='eval')
    logger.log( 'mu=' + str(args.mu) + ' -- Stopping at best acc = ' + str(val_acc1) )

def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    args.mus_list = list(map(float, args.mus_list))
    args.coverage_list = list(map(float, args.coverage_list))
    args.error_list = list(map(float, args.error_list))

    args.class_num = class_num
    logger = prepare_logger(args)

    logger.log("-" * 50)
    logger.log("--- mus = " + str(args.mus_list) )
    logger.log("--- error = " + str(args.error_list) )
    logger.log("--- coverage = " + str(args.coverage_list) )

    if not args.eval:
      for mu in args.mus_list:
        train_one_model_for_mu( args, logger, xshape, mu, train_loader, valid_loader )

    logger.log("-" * 50)
    logger.log('[Evaluation] invoking cov@error with params ')
    evaluate_coverage_error( args, logger, valid_loader, args.eval_batch_size )


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description="Selective Classification(One-Sided Prediction) CIFAR-10/100 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_threshold', type=int, default=1000)
    parser.add_argument('--mus_list', nargs='+', default=[0.5, 1.67])
    parser.add_argument('--error_list', nargs='+', default=[0.005, 0.01, 0.02])
    parser.add_argument('--coverage_list', nargs='+', default=[1., 0.95, 0.90])
    parser.add_argument('--ema-decay', type=float, default=0.996)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--_ckpt', type=str, default='', help="The path to the model checkpoint")
    parser.add_argument("--model_name", type=str, default='ResNet18', help="The path to the model configuration")

    parser.add_argument("--method", type=str, default='OSP', help="The method name. (CE, OSP).")

    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='/home/anilkag/code/compact-vision-nets-PDE-Feature-Generator/data/', help="The dataset name.")
    parser.add_argument("--cutout_length", type=int, default=16, help="The cutout length, negative means not use.")

    # Printing
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=100, help="print frequency (default: 200)")

    parser.add_argument("--save_dir", type=str, help="Folder to save checkpoints and log.", default='./logs/')
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers (default: 8)")

    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=2007, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")

    # Optimization options
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=200, help="Batch size for training.")

    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', help='checkpoint file format')

    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for a single GPU')
    parser.add_argument('--max_lr', type=float, default=0.001, help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"

    main(args)



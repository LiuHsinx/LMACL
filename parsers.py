import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--inter_batch', default=10240, type=int, help='batch size')
    parser.add_argument('--d', default=32, type=int, help='embedding size')
    parser.add_argument('--data', default='tmall', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0, type=float, help='rate for edge dropout')
    parser.add_argument("--num-hidden", type=int, default=32, help="number of hidden units")
    parser.add_argument("--in-drop", type=float, default=0, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0, help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2, help="the negative slope of leaky relu")
    parser.add_argument("--num-layers", type=int, default=1, help="number of gat hidden layers")
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss_s')
    parser.add_argument('--lambda2', default=1e-4, type=float, help='l2 reg weight')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature in cl loss')
    parser.add_argument("--num-heads", type=int, default=6, help="number of hidden attention heads")

    return parser.parse_args()


args = parse_args()

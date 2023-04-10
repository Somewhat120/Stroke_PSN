import dgl
import torch as th
from torch import nn
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.utils import expand_as_pair


class AttnNet(nn.Module):
    """Attention based multi-instance learning layer.
    """

    def __init__(self,
                 in_feats):
        super(AttnNet, self).__init__()
        self.project = nn.Linear(in_feats, in_feats)

    #         self.project = nn.Sequential(nn.Linear(in_feats, 10),
    #                                      nn.Tanh(),
    #                                      nn.Linear(10, 1,
    #                                                bias=False))

    def forward(self, feat, sims):
        attn = th.softmax(self.project(feat), dim=1).unsqueeze(dim=-1)
        fu_sim = th.matmul(sims.T, attn).squeeze(dim=-1)
        return fu_sim, attn.squeeze(dim=-1)


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 dp=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_feats,
                                           int(in_feats / 4)),
                                 nn.Dropout(dp),
                                 nn.Tanh(),
                                 nn.Linear(int(in_feats / 4), 1,
                                           bias=False))

    def forward(self, feat):
        return self.mlp(feat)


class Model(nn.Module):

    def __init__(self,
                 in_feats,
                 hidden_feats,
                 num_heads,
                 dp=0.,
                 sample='no_sp'):
        super(Model, self).__init__()
        self.attnnet = AttnNet(in_feats)
        self.gat1 = GATConv(in_feats,
                            int(hidden_feats / num_heads),
                            num_heads, feat_drop=dp, attn_drop=dp)
        self.gat2 = GATConv(int(hidden_feats / num_heads) * num_heads,
                            int(hidden_feats / num_heads),
                            num_heads, feat_drop=dp, attn_drop=dp)
        self.mlp = MLP(int(hidden_feats / num_heads) * num_heads, dp)
        if sample == 'no_sp':
            self.sample = False
        else:
            self.sample = True

    def forward(self, feat, g):
        # Fuse the multi-PSN based on characteristic-level attention
        # fu_sim, attn = self.attnnet(feat, sims)
        # Construct unweighted homogeneous graph based on Topk filtering
        # _, idx = sims.topk(k, dim=1)
        # dst_ids = idx.flatten()
        # src_ids = th.tensor([i // k for i in range(dst_ids.shape[0])]).to(self.device)
        # g = dgl.graph((src_ids, dst_ids))
        # g = dgl.add_self_loop(g)
        # Learn hidden representations based on GAT
        if not self.sample:
            h = self.gat1(g, feat)
            h = h.flatten(start_dim=1)
            h, attn_gat = self.gat2(g, h, get_attention=True)
            h = h.flatten(start_dim=1)
        else:
            h = self.gat1(g[0], feat)
            h = h.flatten(start_dim=1)
            h, attn_gat = self.gat2(g[1], h, get_attention=True)
            h = h.flatten(start_dim=1)
        # Prediction
        p = self.mlp(h)
        return p, attn_gat
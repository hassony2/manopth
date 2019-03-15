import os

import numpy as np
import torch
from torch.nn import Module

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth import rodrigues_layer, rotproj
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)


class ManoLayer(Module):
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx'
    ]

    def __init__(self,
                 center_idx=None,
                 flat_hand_mean=True,
                 ncomps=6,
                 side='right',
                 mano_root='mano/models',
                 use_pca=True):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
        """
        super().__init__()

        self.center_idx = center_idx
        self.rot = 3
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components']

        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]
                              ) if flat_hand_mean else smpl_data['hands_mean']
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca:
            # Save as axis-angle
            self.register_buffer('th_hands_mean', th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer('th_selected_comps',
                                 torch.Tensor(selected_components))
        else:
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(
                th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer('th_hands_mean_rotmat', th_hands_mean_rotmat)

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self,
                th_pose_coeffs,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1),
                root_palm=torch.Tensor([0])):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """
        # if len(th_pose_coeffs) == 0:
        #     return th_pose_coeffs.new_empty(0), th_pose_coeffs.new_empty(0)

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca:
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                 self.ncomps]
            th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            th_full_pose = torch.cat([
                th_pose_coeffs[:, :self.rot],
                self.th_hands_mean + th_full_hand_pose
            ], 1)
            th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
            th_full_pose = th_full_pose.view(batch_size, -1, 3)
            root_rot = rodrigues_layer.batch_rodrigues(
                th_full_pose[:, 0]).view(batch_size, 3, 3)
        else:
            assert th_pose_coeffs.dim() == 4, (
                'When not self.use_pca, '
                'th_pose_coeffs should have 4 dims, got {}'.format(
                    th_pose_coeffs.dim()))
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_pose_coeffs.shape[2:4]))
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        # Full axis angle representation with root joint
        if th_betas is None or bool(torch.norm(th_betas) == 0):
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       self.th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)

        else:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(15):
            i_val = int(i + 1)
            joint_rot = th_rot_map[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, 16),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(16):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = torch.stack([
                th_verts[:, 745], th_verts[:, 317], th_verts[:, 444],
                th_verts[:, 556], th_verts[:, 673]
            ],
                               dim=1)
        else:
            tips = torch.stack([
                th_verts[:, 745], th_verts[:, 317], th_verts[:, 445],
                th_verts[:, 556], th_verts[:, 673]
            ],
                               dim=1)
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = torch.stack([
            th_jtr[:, 0], th_jtr[:, 13], th_jtr[:, 14], th_jtr[:, 15],
            th_jtr[:, 16], th_jtr[:, 1], th_jtr[:, 2], th_jtr[:, 3],
            th_jtr[:, 17], th_jtr[:, 4], th_jtr[:, 5], th_jtr[:, 6],
            th_jtr[:, 18], th_jtr[:, 10], th_jtr[:, 11], th_jtr[:, 12],
            th_jtr[:, 19], th_jtr[:, 7], th_jtr[:, 8], th_jtr[:, 9],
            th_jtr[:, 20]
        ],
                             dim=1)

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        # Scale to milimeters
        th_verts = th_verts * 1000
        th_jtr = th_jtr * 1000
        return th_verts, th_jtr

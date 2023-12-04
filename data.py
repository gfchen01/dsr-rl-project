import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import h5py
from utils import imretype, draw_arrow


class Data(Dataset):
    def __init__(self, data_path, split, seq_len):
        self.data_path = data_path
        #TODO:
        self.tot_seq_len = seq_len
        self.seq_len = seq_len
        self.volume_size = [128, 128, 48]
        self.direction_num = 8
        self.voxel_size = 0.004
        self.idx_list = open(osp.join(self.data_path, '%s.txt' % split)).read().splitlines()
        self.returns = ['action', 'color_heightmap', 'color_image', 'tsdf', 'mask_3d', 'scene_flow_3d','policy','positions',
        'orientations','new_positions','new_orientations','obj_collision_info','put_positions','put_orientations']
        self.data_per_seq = self.tot_seq_len // self.seq_len

    def __getitem__(self, index):
        data_dict = {}
        idx_seq = index // self.data_per_seq
        idx_step = index % self.data_per_seq * self.seq_len
        for step_id in range(self.seq_len):
            f = h5py.File(osp.join(self.data_path, "%s_%d.hdf5" % (self.idx_list[idx_seq], idx_step + step_id)), "r")

            # action
            action = f['action']
            data_dict['%d-action' % step_id] = self.get_action(action)

            # color_image, [W, H, 3]
            if 'color_image' in self.returns:
                data_dict['%d-color_image' % step_id] = np.asarray(f['color_image_small'], dtype=np.uint8)

            # color_heightmap, [128, 128, 3]
            if 'color_heightmap' in self.returns:
                # draw arrow for visualization
                color_heightmap = draw_arrow(
                    np.asarray(f['color_heightmap'], dtype=np.uint8),
                    (int(action[2]), int(action[1]), int(action[0]))
                )
                data_dict['%d-color_heightmap' % step_id] = color_heightmap

            # tsdf, [S1, S2, S3]
            if 'tsdf' in self.returns:
                data_dict['%d-tsdf' % step_id] = np.asarray(f['tsdf'], dtype=np.float32)

            # mask_3d, [S1, S2, S3]
            if 'mask_3d' in self.returns:
                data_dict['%d-mask_3d' % step_id] = np.asarray(f['mask_3d'], dtype=int)

            # scene_flow_3d, [3, S1, S2, S3]
            if 'scene_flow_3d' in self.returns:
                scene_flow_3d = np.asarray(f['scene_flow_3d'], dtype=np.float32).transpose([3, 0, 1, 2])
                data_dict['%d-scene_flow_3d' % step_id] = scene_flow_3d

            #TODO: kewen add the policies and the pqs
            if 'policy' in self.returns:
                policy = np.asarray(f['policy'], dtype=np.float32)
                data_dict['%d-policy' % step_id] = policy
            if 'positions' in self.returns:
                positions = np.asarray(f['positions'], dtype=np.float32)
                data_dict['%d-positions' % step_id] = positions
            if 'orientations' in self.returns:
                orientations = np.asarray(f['orientations'], dtype=np.float32)
                data_dict['%d-orientations' % step_id] = orientations
            if 'new_positions' in self.returns:
                new_positions = np.asarray(f['new_positions'], dtype=np.float32)
                data_dict['%d-new_positions' % step_id] = new_positions
            if 'new_orientations' in self.returns:
                new_orientations = np.asarray(f['new_orientations'], dtype=np.float32)
                data_dict['%d-new_orientations' % step_id] = new_orientations
            if 'obj_collision_info' in self.returns:
                obj_collision_info = np.asarray(f['obj_collision_info'], dtype=np.float32)
                data_dict['%d-obj_collision_info' % step_id] = obj_collision_info
            if 'put_positions' in self.returns:
                put_positions = np.asarray(f['put_positions'], dtype=np.float32)
                data_dict['%d-put_positions' % step_id] = put_positions
            if 'put_orientations' in self.returns:
                put_orientations = np.asarray(f['put_orientations'], dtype=np.float32)
                data_dict['%d-put_orientations' % step_id] = put_orientations

              
            
            
            

        return data_dict

    def __len__(self):
        return len(self.idx_list) * self.data_per_seq

    def get_action(self, action):
        direction, r, c = int(action[0]), int(action[1]), int(action[2])
        if direction < 0:
            direction += self.direction_num
        action_map = np.zeros(shape=[self.direction_num, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
        action_map[direction, r, c] = 1

        return action_map
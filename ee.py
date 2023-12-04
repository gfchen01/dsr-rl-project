def poke(self,policy):
        # log the current position & quat
        old_po_ors = [p.getBasePositionAndOrientation(object_id) for object_id in self.object_ids]
        output = self.get_scene_info()

        # generate action
        assert policy != None
        policy = policy[0].cpu().numpy()
 
        x_coord, y_coord, z_coord, direction = policy[0],policy[1],policy[2],policy[3]
        #TODO:
        #kewen: record the policy in the dataset
        
        output['policy'] = np.asarray([x_coord, y_coord, z_coord, direction])
        # take action
        self.sim.primitive_push(
            position=[x_coord, y_coord, z_coord],
            rotation_angle=direction / 4.0 * np.pi,
            speed=0.005,
            distance=0.15
        )
        self.sim.robot_go_home()
        #action = {'0': direction, '1': y_pixel, '2': x_pixel}

        mask_3d, scene_flow_3d = self._get_scene_flow_3d(old_po_ors)
        mask_2d, scene_flow_2d = self._get_scene_flow_2d(old_po_ors)
        #TODO:
        #kewen: get the positions and orientations
        positions, orientations = self._get_pos_ori()
        #output['action'] = action
        output['mask_3d'] = mask_3d
        output['scene_flow_3d'] = scene_flow_3d
        output['mask_2d'] = mask_2d
        output['scene_flow_2d'] = scene_flow_2d
        output['new_positions'] = positions
        output['new_orientations'] = orientations
        print('push')
        print(positions)
        print(orientations)
        return output

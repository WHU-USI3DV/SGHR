import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.utils import transform_points
from dataops.dataset import get_dataset_name

class Visual():
    def __init__(self, setname) -> None:
        self.setname = setname
        self.set = get_dataset_name(setname,'./data')

    def __posed_pcds(self,scene):
        n = len(scene.pc_ply_paths)
        pose = np.loadtxt(f'pre/cycle_results/pcposes/{scene.name}/pose.txt',delimiter=',').reshape(n,4,4)
        pcds = []
        for f in range(n):
            pcd = scene.get_pc(f)
            pcd = np.random.permutation(pcd)[0:20000]
            pcd = transform_points(pcd,pose[f])
            pcds.append(pcd)
        return pcds

    def __visual_pcds(self, xyzs, colorize = True, normal = True):
        pcds = []
        for xyz in xyzs:
            if hasattr(xyz,'ndim'):
                xyz = xyz.reshape(-1,3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                if colorize:
                    pcd.paint_uniform_color(np.random.rand(3))
            else: pcd = xyz
            if normal:
                # determine_nei
                points = np.array(pcd.points)
                coor_min = np.amin(points,axis=0)
                coor_max = np.amax(points,axis=0)
                coor_disp = coor_max - coor_min
                coor_disp = np.mean(coor_disp)
                nei = coor_disp / 100.
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(nei, 20))
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)

    def run(self):
        for name, dataset in tqdm(self.set.items()):
            if name == 'wholesetname':continue
            print(f'Visualization of {name}')
            pcds = self.__posed_pcds(dataset)
            self.__visual_pcds(pcds,colorize=False,normal=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='3dmatch',type=str,help='dataset name')
    args = parser.parse_args()
    visualizer = Visual(args.dataset)
    visualizer.run()
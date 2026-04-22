import open3d as o3d
import numpy as np


class PainterO3d:
    def __init__(self, bins):
        ''' '''
        self.itemlist = bins.get_all_items()
        self.width = bins.width
        self.height = bins.height
        self.depth = bins.depth

    def _plotBottom(self, x, y, z, dx, dy, dz, color=[1, 1, 1]):
        """ Auxiliary function to plot the bottom face of a cube using Open3D. """
        box = o3d.geometry.TriangleMesh()
        vertices = np.array([
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z]
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        box.vertices = o3d.utility.Vector3dVector(vertices)
        box.triangles = o3d.utility.Vector3iVector(triangles)
        box.paint_uniform_color(color)
        return box

    def _plotCube(self, x, y, z, dx, dy, dz, color=[1, 1, 1, 0.5], edge_color=[0, 0, 1], i =0):
        """ Auxiliary function to plot a cube using Open3D. """
        box = o3d.geometry.TriangleMesh.create_box(width=dx, height=dy, depth=dz)
        box.translate((x, y, z))
        
        # 원래 색깔로 표시
        # box.paint_uniform_color(color[:3])

        # 순서에 따라 표시
        if i == 0:
            box.paint_uniform_color([1,0,0])
            
        elif i == '_':
            box.paint_uniform_color(color[:3])
        else:
            box.paint_uniform_color([0,1,1])
        edges = self._create_edges(x, y, z, dx, dy, dz, edge_color)
        return box, edges

    def _create_edges(self, x, y, z, dx, dy, dz, color=[0, 0, 1]):
        """ Auxiliary function to create the edges of a cube for visualization using Open3D. """
        points = [
            [x, y, z],
            [x + dx, y, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y + dy, z],
            [x + dx, y, z + dz],
            [x, y + dy, z + dz],
            [x + dx, y + dy, z + dz]
        ]

        lines = [
            [0, 1], [0, 2], [0, 3],
            [1, 4], [1, 5],
            [2, 4], [2, 6],
            [3, 5], [3, 6],
            [4, 7],
            [5, 7],
            [6, 7]
        ]

        colors = [color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def _plotCylinder(self, x, y, z, dx, dy, dz, color=[1, 0, 0]):
        """ Auxiliary function to plot a cylinder using Open3D. """
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=dx / 2, height=dz)
        cylinder.translate((x + dx / 2, y + dy / 2, z))
        cylinder.paint_uniform_color(color)
        return cylinder

    def plotBoxAndItems(self, title="", write_num=False, fontsize=10):
        """ Plot the Bin and the items it contains using Open3D. """
        mesh_list = []
        edge_list = []

        # Plot bin bottom with blue edges
        bottom = self._plotBottom(0, 0, 0, self.width, self.height, 0, color=[1, 1, 1])
        mesh_list.append(bottom)
        edges = self._create_edges(0, 0, 0, self.width, self.height, 0, color=[0, 0, 1])
        edge_list.append(edges)

        # Plot items
        for item in self.itemlist:
            x, y, z = item.position
            w, h, d = item.width, item.height, item.depth
            color = [int(item.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] + [0.5]

            if item.objshape == 'cube':
                box, edges = self._plotCube(x, y, z, w, h, d, color=color, i = item.partno)
                mesh_list.append(box)
                edge_list.append(edges)
            elif item.objshape == 'cylinder':
                mesh_list.append(self._plotCylinder(x, y, z, w, h, d, color=color))

        # Create a visualizer and add geometries
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        for mesh in mesh_list:
            vis.add_geometry(mesh)
        for edge in edge_list:
            vis.add_geometry(edge)

        # Set camera parameters
        ctr = vis.get_view_control()
        ctr.set_zoom(1.8)
        ctr.set_front([0.5, -0.5, 1.0])
        ctr.set_lookat([self.width / 2, self.height / 2, 0])
        ctr.set_up([0.0, 0.0, 1.0])

        # Visualize
        vis.run()
        vis.destroy_window()
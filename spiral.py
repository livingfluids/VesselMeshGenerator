"""
Helical / Spiral Vessel Mesh Generator

Copyright (c) 2025 Leo Liu, FAMU‑FSU College of Engineering
Licensed under the MIT License - see the LICENSE file in the project root.

If you use this code in research, please cite the project and contact the
author: Leo Liu <leo.liu@eng.famu.fsu.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VesselMesh:
    def __init__(self,  # initialize with parameters
                 radius=5.0,           # Base radius of the helix
                 cross_section_radius=0.5,  # Radius of vessel cross-section
                 pitch=5.0,            # Vertical distance per full revolution
                 num_turns=3.0,        # Number of complete helical turns
                 num_points_per_turn=32,  # Resolution around circumference
                 num_turns_segments=64,   # Resolution along helical path
                 mode='helical',       # 'helical' or 'spiral'
                 radius_variation=0.0, # decrease of radius along length for spiral artery
                 taper_ratio=0.0):     # fraction to taper cross-section along length (0..1)

        self.radius = radius
        self.cross_section_radius = cross_section_radius
        self.pitch = pitch
        self.num_turns = num_turns
        self.num_points_per_turn = num_points_per_turn
        self.num_turns_segments = num_turns_segments
        # Spiral-artery related
        self.mode = mode
        self.radius_variation = radius_variation
        self.taper_ratio = taper_ratio
        
        self.vertices = None
        self.triangles = None
        
    def generate_mesh(self):
        """Generate the helical vessel mesh"""
        # Parameters for helical path
        total_angle = 2 * np.pi * self.num_turns
        t_values = np.linspace(0, total_angle, self.num_turns_segments) 
        
        # Angular positions around circumference
        theta_values = np.linspace(0, 2 * np.pi, self.num_points_per_turn, endpoint=False)
        
        vertices = []
        
        # Generate vertices for the outer surface only
        for i, t in enumerate(t_values):
            # Compute fraction along the helix using angle t (so radius changes along revolutions)
            frac_along = t / total_angle

            # Compute current centerline radius depending on mode
            current_radius = self.radius
            if self.mode == 'spiral':  # spiral_artery: radius changes along revolutions (linear by default)
                current_radius = self.radius - self.radius_variation * frac_along
                # print(f"t={t:.2f}, frac_along={frac_along:.3f}, current_radius={current_radius:.3f}")

            # Helical centerline
            center_x = current_radius * np.cos(t)
            center_y = current_radius * np.sin(t)
            center_z = (self.pitch * t) / (2 * np.pi)

            # Local coordinate system for cross-section
            tangent = np.array([-current_radius * np.sin(t),
                                current_radius * np.cos(t),
                                self.pitch / (2 * np.pi)])
            tangent = tangent / np.linalg.norm(tangent)

            normal = np.array([-np.cos(t), -np.sin(t), 0])

            binormal = np.cross(tangent, normal)
            binormal = binormal / np.linalg.norm(binormal)

            # Generate circular cross-section (apply taper if needed)
            csr = self.cross_section_radius * max(0.001, (1.0 - self.taper_ratio * frac_along)) # 0.001 to avoid zero radius

            for theta in theta_values:
                local_x = csr * np.cos(theta)
                local_y = csr * np.sin(theta)

                # Transform to global coordinates
                point = np.array([center_x, center_y, center_z]) + \
                        local_x * normal + local_y * binormal

                vertices.append(point)
        
        self.vertices = np.array(vertices)
        
        # Generate triangular faces
        triangles = []
        
        # Number of vertices for the single outer surface
        verts_total = self.num_turns_segments * self.num_points_per_turn

        # Generate triangles across the surface grid
        for i in range(self.num_turns_segments - 1):
            for j in range(self.num_points_per_turn):
                v1 = i * self.num_points_per_turn + j
                v2 = i * self.num_points_per_turn + (j + 1) % self.num_points_per_turn
                v3 = (i + 1) * self.num_points_per_turn + j
                v4 = (i + 1) * self.num_points_per_turn + (j + 1) % self.num_points_per_turn

                # Two triangles per quad (consistent winding)
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
        
        # No caps or inner surface — open helical tube (outer surface only)
        
        self.triangles = np.array(triangles)
        
        return self.vertices, self.triangles
    
    def visualize(self, show_wireframe=True, show_surface=True,
                  wireframe_stride=1, wireframe_linewidth=1.0,
                  wireframe_color='k', wireframe_alpha=0.8):
        """Visualize the generated mesh"""
        if self.vertices is None or self.triangles is None:
            self.generate_mesh()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_surface:
            # Plot triangular mesh
            ax.plot_trisurf(self.vertices[:, 0], 
                           self.vertices[:, 1], 
                           self.vertices[:, 2],
                           triangles=self.triangles,
                           alpha=0.7, 
                           cmap='viridis')
        
        if show_wireframe:
            # Plot wireframe (use configurable styling)
            for triangle in self.triangles[::wireframe_stride]:  # Show every Nth triangle to avoid clutter
                pts = self.vertices[triangle]
                # Close the triangle
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=wireframe_color,
                        alpha=wireframe_alpha, linewidth=wireframe_linewidth)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Vessel Mesh\n{len(self.vertices)} vertices, {len(self.triangles)} triangles')
        
        # Equal aspect ratio
        max_range = np.array([self.vertices[:,0].max()-self.vertices[:,0].min(),
                             self.vertices[:,1].max()-self.vertices[:,1].min(),
                             self.vertices[:,2].max()-self.vertices[:,2].min()]).max() / 2.0
        mid_x = (self.vertices[:,0].max()+self.vertices[:,0].min()) * 0.5
        mid_y = (self.vertices[:,1].max()+self.vertices[:,1].min()) * 0.5
        mid_z = (self.vertices[:,2].max()+self.vertices[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def export_obj(self, filename):
        """Export mesh to OBJ format"""
        if self.vertices is None or self.triangles is None:
            self.generate_mesh()
        
        with open(filename, 'w') as f:
            f.write(f"# Vessel mesh\n")
            f.write(f"# {len(self.vertices)} vertices, {len(self.triangles)} faces\n\n")
            
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for tri in self.triangles:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        
        print(f"Mesh exported to {filename}")
    
    def export_stl_ascii(self, filename):
        """Export mesh to ASCII STL format"""
        if self.vertices is None or self.triangles is None:
            self.generate_mesh()
        
        with open(filename, 'w') as f:
            f.write(f"solid vessel\n")
            
            for tri in self.triangles:
                # Calculate normal vector
                v1, v2, v3 = self.vertices[tri]
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write(f"    outer loop\n")
                for vertex_idx in tri:
                    v = self.vertices[vertex_idx]
                    f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write(f"endsolid vessel\n")
        
        print(f"Mesh exported to {filename}")
    
    def get_mesh_info(self):
        """Get information about the generated mesh"""
        if self.vertices is None or self.triangles is None:
            self.generate_mesh()
        
        info = { # Mesh statistics in a dictionary
            'num_vertices': len(self.vertices),
            'num_triangles': len(self.triangles),
            'bounding_box': { # Axis-aligned bounding box with min/max coordinates
                'min': self.vertices.min(axis=0),
                'max': self.vertices.max(axis=0)
            },
            'parameters': { # dic inside another dic
                'radius': self.radius,
                'pitch': self.pitch,
                'num_turns': self.num_turns,
                'resolution_circumferential': self.num_points_per_turn,
                'resolution_helical': self.num_turns_segments,
                'mode': self.mode,
                'radius_variation': self.radius_variation,
                'taper_ratio': self.taper_ratio
            }
        }
        
        return info


# main execution
if __name__ == "__main__":

    # --- Helical artery --- #
    print("Creating a helical arteriole mesh...")
    helical = VesselMesh(
        radius=100.0,           # Base radius of the helix
        cross_section_radius=15.0,  # Radius of vessel cross-section
        pitch=100.0,            # Vertical distance per full revolution
        num_turns=6.0,        # Number of complete helical turns
        num_points_per_turn=20,  # Resolution around circumference
        num_turns_segments=100,   # Resolution along helical path
        mode='helical'        
    )
    
    # Generate mesh
    vertices, triangles = helical.generate_mesh()
    
    # Print mesh information
    info = helical.get_mesh_info()
    print("Mesh Information:")
    print(f"Vertices: {info['num_vertices']}")
    print(f"Triangles: {info['num_triangles']}")
    print(f"Bounding box: {info['bounding_box']['min']} to {info['bounding_box']['max']}")
    
    # Visualize the mesh
    helical.visualize(show_wireframe=True, show_surface=True)
    
    # Export to different formats
    helical.export_obj("helical_vessel.obj")
    helical.export_stl_ascii("helical_vessel.stl")    

    # --- Spiral artery --- #
    print('\nCreating a spiral arteriole mesh...')
    spiral = VesselMesh(
        radius=100.0,           # Base radius of the helix
        cross_section_radius=15.0,  # Radius of vessel cross-section
        pitch=100.0,            # Vertical distance per full revolution
        num_turns=6.0,        # Number of complete helical turns
        num_points_per_turn=20,  # Resolution around circumference
        num_turns_segments=100,   # Resolution along helical path
        mode='spiral',        
        radius_variation=50.0,
        taper_ratio=0.35
    )
    a_verts, a_tris = spiral.generate_mesh()
    a_info = spiral.get_mesh_info()
    
    # Print mesh information
    info = spiral.get_mesh_info()
    print("Mesh Information:")
    print(f"Vertices: {info['num_vertices']}")
    print(f"Triangles: {info['num_triangles']}")
    print(f"Bounding box: {info['bounding_box']['min']} to {info['bounding_box']['max']}")

    # Visualize spiral artery
    spiral.visualize(show_wireframe=True, show_surface=True)
    
    # Export to different formats
    spiral.export_obj("spiral_artery.obj")
    spiral.export_stl_ascii("spiral_artery.stl")
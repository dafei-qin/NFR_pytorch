import os
import numpy as np
from struct import pack, unpack
from copy import deepcopy
import trimesh
"""
Reads OBJ files
Only handles vertices, faces and UV maps
Input:
- file: path to .obj file
Outputs:
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data in .obj file, it shall return Vt=None and Ft=None
"""

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.edges = None
        self.vertex_degree = None
        self.face_adjacency_edges = None
        self.face_adjacency_unshared = None
        # self.edges_unique_length = None
        self.edges_unique = None
        # self.area_faces = None

        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)



    def __getitem__(self, key):
        if key == 0:
            return self.vertices

        elif key == 1:
            return self.faces
        else:
            raise KeyError # print(f"Only allows Key 0: vertices, Key 1: faces, but receives Key {key} ")

    def update_area(self):
        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)


    @classmethod
    def load(self, path: str, read_face=True):
        """
        Load obj file
        load the .obj format mesh file with square or triangle faces
        return the vertices list and faces list
        """
        if path.endswith('.obj'):
            file = open(path, 'r')
            lines = file.readlines()
            vertices = []
            faces = []
            for line in lines:
                if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                    line_split = line.split(" ")
                    # ver = line_split[1:4]
                    ver = [each for each in line_split[1:] if each != '']
                    ver = [float(v) for v in ver]
                    vertices.append(ver)
                else:
                    if read_face:
                        if line.startswith('f'):
                            line_split = line.split(" ")
                            if '/' in line:
                                tmp_faces = line_split[1:]
                                f = []
                                if '\n' in tmp_faces:
                                    tmp_faces.pop(tmp_faces.index('\n'))
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face.split('/')[0]))
                                faces.append(f)
                            else:
                                tmp_faces = line_split[1:]
                                f = []
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face))
                                faces.append(f)
                    else:
                        pass

            if read_face:
                file.close()
                return Mesh(np.array(vertices), np.array(faces) - 1)
            else:
                file.close()
                return Mesh(np.array(vertices), None)
        # else:
        #     raise ValueError('Wrong file format, not a correct .obj mesh file!')
        #     ret

            
    @classmethod
    def from_trimesh(self, mesh: trimesh.Trimesh):
        
        new_mesh = Mesh(deepcopy(mesh.vertices), deepcopy(mesh.faces))

        new_mesh.edges = deepcopy(mesh.edges)
        new_mesh.vertex_degree = deepcopy(mesh.vertex_degree)
        new_mesh.face_adjacency_edges = deepcopy(mesh.face_adjacency_edges)
        new_mesh.face_adjacency_unshared = deepcopy(mesh.face_adjacency_unshared)
        # new_mesh.edges_unique_length = deepcopy(mesh.edges_unique_length)
        new_mesh.edges_unique = deepcopy(mesh.edges_unique)
        # new_mesh.area_faces = deepcopy(mesh.area_faces)
        
        return new_mesh


    
    def transfer(self, shift, scale):
        self.vertices = self.vertices * np.array(scale)[np.newaxis]
        self.vertices = self.vertices + np.array(shift)[np.newaxis]
        self.update_area()
        
    def write(self, file_name_path):
        faces = self.faces
        vertices = self.vertices
        faces = faces + 1
        with open(file_name_path, 'w') as f:
            for v in vertices:
                # print(v)
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for face in faces:
                if len(face) == 4:
                    f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])) 
                if len(face) == 3:
                    f.write("f {} {} {}\n".format(face[0], face[1], face[2])) 
                    

def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, "r") as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith("v "):
            v = [float(n) for n in t.replace("v ", "").split(" ")]
            V += [v]
        # UV vertex
        elif t.startswith("vt "):
            v = [float(n) for n in t.replace("vt ", "").split(" ")]
            Vt += [v]
        # Face
        elif t.startswith("f "):
            idx = [n.split("/") for n in t.replace("f ", "").split(" ")]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if "/" in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft:
        assert len(F) == len(
            Ft
        ), "Inconsistent .obj file, mesh and UV map do not have the same number of faces"
    else:
        Vt, Ft = None, None
    return V, F, Vt, Ft


"""
Writes OBJ files
Only handles vertices, faces and UV maps
Inputs:
- file: path to .obj file (overwrites if exists)
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data as input, it will write only 3D data in .obj file
"""


def writeOBJ(file, V, F, Vt=None, Ft=None):
    if not Vt is None:
        assert len(F) == len(
            Ft
        ), "Inconsistent data, mesh and UV map do not have the same number of faces"

    with open(file, "w") as file:
        # Vertices
        for v in V:
            line = "v " + " ".join([str(_) for _ in v]) + "\n"
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = "vt " + " ".join([str(_) for _ in v]) + "\n"
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [
                [str(i + 1) + "/" + str(j + 1) for i, j in zip(f, ft)]
                for f, ft in zip(F, Ft)
            ]
        else:
            F = [[str(i + 1) for i in f] for f in F]
        for f in F:
            line = "f " + " ".join(f) + "\n"
            file.write(line)


"""
Reads PC2 files, and proposed format PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- float16: False for PC2 files, True for PC16
Output:
- data: dictionary with .pc2/.pc16 file data
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def readPC2(file, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    data = {}
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file, "rb") as f:
        # Header
        data["sign"] = f.read(12)
        # data['version'] = int.from_bytes(f.read(4), 'little')
        data["version"] = unpack("<i", f.read(4))[0]
        # Num points
        # data['nPoints'] = int.from_bytes(f.read(4), 'little')
        data["nPoints"] = unpack("<i", f.read(4))[0]
        # Start frame
        data["startFrame"] = unpack("f", f.read(4))
        # Sample rate
        data["sampleRate"] = unpack("f", f.read(4))
        # Number of samples
        # data['nSamples'] = int.from_bytes(f.read(4), 'little')
        data["nSamples"] = unpack("<i", f.read(4))[0]
        # Animation data
        size = data["nPoints"] * data["nSamples"] * 3 * bytes
        data["V"] = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
        data["V"] = data["V"].reshape(data["nSamples"], data["nPoints"], 3)

    return data


"""
Reads an specific frame of PC2/PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- frame: number of the frame to read
- float16: False for PC2 files, True for PC16
Output:
- T: mesh vertex data at specified frame
"""


def readPC2Frame(file, frame, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    assert frame >= 0 and isinstance(frame, int), "Frame must be a positive integer"
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file, "rb") as f:
        # Num points
        f.seek(16)
        # nPoints = int.from_bytes(f.read(4), 'little')
        nPoints = unpack("<i", f.read(4))[0]
        # Number of samples
        f.seek(28)
        # nSamples = int.from_bytes(f.read(4), 'little')
        nSamples = unpack("<i", f.read(4))[0]
        if frame > nSamples:
            print("Frame index outside size")
            print("\tN. frame: " + str(frame))
            print("\tN. samples: " + str(nSamples))
            return
        # Read frame
        size = nPoints * 3 * bytes
        f.seek(size * frame, 1)  # offset from current '1'
        T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
    return T.reshape(nPoints, 3)


"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    if float16:
        V = V.astype(np.float16)
    else:
        V = V.astype(np.float32)
    with open(file, "wb") as f:
        # Create the header
        headerFormat = "<12siiffi"
        headerStr = pack(
            headerFormat, b"POINTCACHE2\0", 1, V.shape[1], 0, 1, V.shape[0]
        )
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())


"""
Appends frames to PC2 and PC16 files
Inputs:
- file: path to file
- V: 3D animation data as a three dimensional array (N. New Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2Frames(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    # Read file metadata (dimensions)
    if os.path.isfile(file):
        if float16:
            V = V.astype(np.float16)
        else:
            V = V.astype(np.float32)
        with open(file, "rb+") as f:
            # Num points
            f.seek(16)
            nPoints = unpack("<i", f.read(4))[0]
            assert len(V.shape) == 3 and V.shape[1] == nPoints, (
                "Inconsistent dimensions: "
                + str(V.shape)
                + " and should be (-1,"
                + str(nPoints)
                + ",3)"
            )
            # Read n. of samples
            f.seek(28)
            nSamples = unpack("<i", f.read(4))[0]
            # Update n. of samples
            nSamples += V.shape[0]
            f.seek(28)
            f.write(pack("i", nSamples))
            # Append new frame/s
            f.seek(0, 2)
            f.write(V.tobytes())
    else:
        writePC2(file, V, float16)
        

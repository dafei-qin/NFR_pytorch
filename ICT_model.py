import sys
import numpy as np
import torch
from myutils import Mesh
import time
import logging
sys.path.append('./third_party/ICT-FaceKit/Scripts/')
try:
    import face_model_io
    has_ICT_orig = True

except ModuleNotFoundError:
    has_ICT_orig = False

class ICT_model():
    # Customized ICT model with faster loading speed.
    def __init__(self, path, un_subdivide=False, device='cpu', load=None, code=torch.zeros((1, 153))):
        self.code = code 
        if load is not None:

            return self.load(load)
        if not has_ICT_orig:
            logging.error('No cached model found, please use the ICT_Face_Kit to generate the cache first')
            
        face_model =  face_model_io.load_face_model(path)
        referenced = np.load('./third_party/ICT-FaceKit/11248-11089.npy')
        self.vertices = torch.from_numpy(face_model._deformed_vertices).to(device)[:11248][referenced]
        self.identity_names = face_model._identity_names
        self.expression_names = face_model._expression_names
        self.identity_weights = torch.from_numpy(face_model._identity_shape_modes)[:, :11248][:, referenced]
        self.expression_weights = torch.from_numpy(face_model._expression_shape_modes)[:, :11248][:, referenced]
        self.weights = torch.cat((self.identity_weights, self.expression_weights), dim=0)
        if un_subdivide:
            un_subdivide_mapping = np.load('./third_party/ICT-FaceKit/10089-3694.npy')
            self.vertices = self.vertices[un_subdivide_mapping]
            self.identity_weights = self.identity_weights[:, un_subdivide_mapping]
            self.expression_weights = self.expression_weights[:, un_subdivide_mapping]
            self.weights = self.weights[:, un_subdivide_mapping]
            self.faces = Mesh.load('./third_party/ICT-FaceKit/template_only_face_3694.obj').faces
        else:
            self.faces = Mesh.load('./third_party/ICT-FaceKit/template_only_face_10089.obj').faces



    def deform(self, code):
        if type(code) == np.ndarray:
            code = torch.from_numpy(code)
        code = code.cpu()
        # code: (B, 53+100)
        B = code.shape[0]
        if code.shape[1] == 53: # Change the expression
            _code = torch.concat([self.code[:, :100].expand(B, -1), code], dim=-1)
        elif code.shape[1] == 100: # Change the identity
            _code = torch.concat([code, self.code[:, 100:].expand(B, -1)], dim=-1)
        else: # Change both
            _code = code
        t = time.time()
        out_v = self.vertices.unsqueeze(0).expand(B, -1, -1).clone() * 10 # (B, V, 3)
        out_v += (_code.unsqueeze(-1).unsqueeze(-1) * self.weights.unsqueeze(0)).sum(axis=1) # (B, D, 1, 1) x (1, D, V, 3) --> (B, V, 3)
        logging.debug(f'ICT generation: Total time: {time.time() - t:.2f}s | Time per mesh: {(time.time() - t) / B * 1000:.2f}ms')
        out_v = out_v * 0.1
        out_v -= out_v.mean(axis=(0, 1))
        return out_v
    
    def save(self, save_name):
        torch.save({'faces': self.faces, 'vertices': self.vertices, 'identity_weights': self.identity_weights, 'expression_weights': self.expression_weights, 'weights': self.weights, 'identity_names': self.identity_names, 'expression_names': self.expression_names}, save_name)

    @classmethod
    def load(self, type):
        if type == 1:
            load_name = './third_party/ICT-FaceKit/11089.th'
        elif type == 2:
            load_name = './third_party/ICT-FaceKit/3694.th'
        else:
            logging.error(f'Only support type in [1, 2], but found {type}')
            raise NotImplementedError
        obj = self.__new__(self)
        self.code = torch.zeros((1,153))
        obj.faces, obj.vertices, obj.identity_weights, obj.expression_weights, obj.weights, obj.identity_names, obj.expression_names = torch.load(load_name).values()
        obj.vertices *= 0.1
        return obj





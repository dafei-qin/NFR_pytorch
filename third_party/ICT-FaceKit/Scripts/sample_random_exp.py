

"""Example script that samples and writes random identities.
"""
import numpy as np
import face_model_io
import os
from tqdm import tqdm
from copy import deepcopy
def main():
    """Loads the ICT Face Mode and samples and writes 10 random identities.
    """
    # Create a new FaceModel and load the model
    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    job = 'single_random'
    save_dir = "D:\\ICT_data_single_random\\"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # face_model_io.write_deformed_mesh('../sample_data_out/ref.obj', face_model)
    
    # generate expressions
    identity_weights_list = []
    exp_weights_list = [np.zeros((53, 1))]
    for j in range(5):
        identity_weights = np.random.normal(size=100)
        identity_weights_list.append(identity_weights[np.newaxis])
    if job == 'single':
        for i in range(53):
            exp_weights = np.zeros((53, 6))
            exp_weights[i] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            exp_weights_list.append(exp_weights)
            
        # for j in range(i, 53):
        #     if j != i:
        #         for k in [0.4,0.8]:
        #             exp_weights = np.zeros((53, 2))
        #             exp_weights[i] = k
        #             exp_weights[j] = [0.4, 0.8]
        #             exp_weights_list.append(exp_weights)
    elif job == 'random':
        for i in range(5000):
            exp_weights = np.abs(np.random.rand(53, 1) )
            exp_weights[np.random.choice(53, 50), :] = 0
            exp_weights_list.append(exp_weights)
    elif job == 'single_random':
        for i in range(53):
            exp_weights = np.zeros((53, 6))
            exp_weights[i] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            exp_weights_list.append(exp_weights)
        for i in range(5000):
            exp_weights = np.abs(np.random.rand(53, 1) )
            exp_weights[np.random.choice(53, 50, replace=False), :] = 0
            exp_weights_list.append(exp_weights)
        for i in range(500):
            exp_weights = np.abs(np.random.rand(53, 1) )
            exp_weights[np.random.choice(53, 50, replace=False), :] = 0
            exp_weights[26] = 1.5
            exp_weights_list.append(exp_weights)
    identity_weights_list = np.concatenate(identity_weights_list, axis=0)
    exp_weights_list = np.concatenate(exp_weights_list, axis=-1).T
    np.savez(os.path.join(save_dir, f'./iden_exp_weights_single.npz'), iden=identity_weights_list, exp=exp_weights_list)
    # identity_weights_list, exp_weights_list = np.load('./iden_exp_weights_single.npz').values()

    for idx, iden_vec in enumerate(tqdm(identity_weights_list)):
        # current_save_dir = os.path.join(save_dir, f'{idx:03d}')
        # if not os.path.exists(current_save_dir):
        #     os.makedirs(current_save_dir, exist_ok=True)
        face_model._identity_weights = iden_vec
        face_model._expression_weights = np.zeros(53)
        face_model.deform_mesh()
        vertices = face_model._deformed_vertices[:11248]
        np.save(os.path.join(save_dir, f'{idx:03d}_neutral.npy'), vertices.astype(np.float32))
        v_store = []
        for idxx, exp_vec in enumerate(exp_weights_list):
            face_model.set_expression(exp_vec)
            face_model.deform_mesh()
            vertices = face_model._deformed_vertices[:11248]
            v_store.append(deepcopy(vertices)[np.newaxis])
        v_store = np.concatenate(v_store, axis=0)

        np.save(os.path.join(save_dir, f'{idx:03d}.npy'), v_store.astype(np.float32))


            # face_model_io.write_deformed_mesh(write_path, face_model)
    # for i in range(150):
    #     # Set the identity
    #     face_model._identity_weights = identity_weights_list[i]
    #     face_model.deform_mesh()
        
    #     # Randomize the identity and deform the mesh
    #     #face_model.randomize_identity()
    #     exp_coeffs = np.random.rand(exp_coeffs.shape[0]) * 0.5
    #     face_model.set_expression(exp_coeffs)
    #     face_model.deform_mesh()

    #     # Write the deformed mesh
    #     write_path = '../generated_training_set/random_expression{:02d}.obj'.format(i)
    #     face_model_io.write_deformed_mesh(write_path, face_model)

    # print("Finished writing meshes.")
    

if __name__ == '__main__':
    main()

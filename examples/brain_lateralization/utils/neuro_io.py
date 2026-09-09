import numpy as np


def read_surf(surface):
    import nibabel as nib

    surf = nib.load(surface)
    arr = surf.darrays
    coord = arr[0].data
    vert = arr[1].data
    return coord, vert


def read_nii(T1w):
    import nibabel as nib

    T1 = nib.load(T1w)
    data = T1.get_fdata()
    header = T1.header
    return data, header


def read_shape_gii(shape_gii):
    import nibabel as nib

    gii = nib.load(shape_gii)
    data = gii.darrays[0].data
    return gii, data


def create_shape_gii(shape_gii_base, array_write, savepath):
    import nibabel as nib

    gii, _ = read_shape_gii(shape_gii_base)
    gii.darrays[0].data = array_write
    nib.save(gii, savepath)


def create_fs_lr32k_atlas(shape_gii_base, array_write, atlas_file, hemi, savepath):
    import nibabel as nib

    f_atlas = nib.load(atlas_file)
    f_data = f_atlas.get_fdata()[0, :]

    if hemi == "L":
        cdata = f_data[0 : int(len(f_data) / 2)].copy()
        print("cdata.shape: %d" % (len(cdata)))
        for i, data in enumerate(array_write):
            cdata[np.where(cdata == i + 1)] = data
    else:
        cdata = f_data[int(len(f_data) / 2) : int(len(f_data))].copy()
        for i, data in enumerate(array_write):
            cdata[np.where(cdata == i + 1 + int(np.max(f_data) / 2))] = data

    create_shape_gii(shape_gii_base, cdata, savepath)


def creat_shape_gii(shape_gii_base, array_write, savepath):
    return create_shape_gii(shape_gii_base, array_write, savepath)


def creat_fs_lr32k_atlas(shape_gii_base, array_write, atlas_file, hemi, savepath):
    return create_fs_lr32k_atlas(shape_gii_base, array_write, atlas_file, hemi, savepath)

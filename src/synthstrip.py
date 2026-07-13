import torch
import torch.nn as nn
import numpy as np
import surfa as sf
import nibabel as nib
import scipy.ndimage


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class StripModel(nn.Module):

    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):

        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
    
    
    
class CustomSynthstrip:
    
    def __init__(self, gpu:bool):
        # necessary for speed gains (I think)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
        if gpu:
            if not torch.cuda.is_available(): print('GPU asked for SynthStrip but CUDA is not available')
            self.device = torch.device('cuda')
            device_name = 'GPU'
        else:
            self.device = torch.device('cpu')
            device_name = 'CPU'
        print(f'Configuring model on the {device_name}')
        
        with torch.no_grad():
            self.model = StripModel()
            self.model.to(self.device)
            self.model.eval()
            
        modelfile = '/home/vincent/Downloads/synthstrip.1.pt'
        checkpoint = torch.load(modelfile, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.border = 1

    def skullstrip(
        self,
        mri:nib.nifti1.Nifti1Image
    ) -> tuple[nib.nifti1.Nifti1Image,nib.nifti1.Nifti1Image]:
        geom = sf.ImageGeometry(shape=mri.shape[:3], vox2world=mri.affine)
        image = sf.Volume(mri.get_fdata(), geometry=geom)
        
        # conform image and fit to shape with factors of 64
        conformed = image.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA').crop_to_bbox()
        target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
        conformed = conformed.reshape(target_shape)
        
        # normalize intensities
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)
        
        # predict the surface distance transform
        with torch.no_grad():
            input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(self.device)
            sdt = self.model(input_tensor).cpu().numpy().squeeze()
            
        # unconform the sdt and extract mask
        sdt = conformed.new(sdt).resample_like(image, fill=100)
        
        # find largest CC (just do this to be safe for now)
        components = scipy.ndimage.label(sdt.data < self.border)[0]
        bincount = np.bincount(components.flatten())[1:]
        mask = (components == (np.argmax(bincount) + 1))
        mask = scipy.ndimage.binary_fill_holes(mask)
        image[~mask] = 0
        
        return (
            nib.Nifti1Image(image.data, affine=mri.affine, header=mri.header), 
            nib.Nifti1Image(mask.astype('int'), affine=mri.affine, header=mri.header)
        )
import torch
import torch.nn as nn

from copy import deepcopy
from models.dust3r.utils.misc import is_symmetrized, interleave
from models.dust3r.utils.heads import head_factory
from models.dust3r.utils.misc import transpose_to_landscape
from .model import AsymmetricCroCo3DStereo

class DynaDust3r(AsymmetricCroCo3DStereo):
    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', float('-inf'), float('inf')),
                 conf_mode=('exp', 1, float('inf')),
                 motion_mode='flow3d',
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',
                 **croco_kwargs):
        super().__init__(
            output_mode=output_mode,
            head_type=head_type,
            depth_mode=depth_mode,
            conf_mode=conf_mode,
            freeze=freeze,
            landscape_only=landscape_only,
            patch_embed_cls=patch_embed_cls,
            **croco_kwargs
        )
        self.set_motion_heads(motion_mode, landscape_only)

    def set_motion_heads(self, motion_mode, landscape_only):
        self.motion_head1 = head_factory(self.head_type, motion_mode, self, has_conf=True)
        self.motion_head2 = head_factory(self.head_type, motion_mode, self, has_conf=True)
        self.mhead1 = transpose_to_landscape(self.motion_head1, activate=landscape_only)
        self.mhead2 = transpose_to_landscape(self.motion_head2, activate=landscape_only)

    def forward(self, view1, view2, t_query=1.0):
        img1, img2 = view1['img'], view2['img']
        B = img1.shape[0]
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B,1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B,1))

        if is_symmetrized(view1, view2):
            f1, f2, p1, p2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            f1, f2 = interleave(f1, f2)
            p1, p2 = interleave(p1, p2)
        else:
            f1, f2, p1, p2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        d1, d2 = self._decoder(f1, p1, f2, p2)
        with torch.amp.autocast(enabled=False):
            dec1 = [tok.float() for tok in d1]
            dec2 = [tok.float() for tok in d2]
            r1 = self._downstream_head(1, dec1, shape1)
            r2 = self._downstream_head(2, dec2, shape2)
            m1 = self._motion_head(self.mhead1, dec1, shape1, t_query)
            m2 = self._motion_head(self.mhead2, dec2, shape2, t_query)

        r1['motion'] = m1
        r2['motion'] = m2
        r2['pts3d_in_other_view'] = r2.pop('pts3d')
        return r1, r2

    def _motion_head(self, head, dec_out, img_shape, t_query):
        return head(dec_out, img_shape, t_query=t_query)

from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion

from .pv_rcnn_ssl import PVRCNN_SSL
from .pv_rcnn_ssl_naive import PVRCNN_SSL_Naive
from .pv_rcnn_ssl_progt import PVRCNN_SSL_ProGT
from .pv_rcnn_ssl_cer import PVRCNN_SSL_CER 
from .pv_rcnn_ssl_consis import PVRCNN_SSL_Consis

from .pv_rcnn_pseudo import PVRCNN_Pseudo

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,

    'PVRCNN_SSL': PVRCNN_SSL,
    'PVRCNN_SSL_Naive': PVRCNN_SSL_Naive,
    'PVRCNN_SSL_ProGT': PVRCNN_SSL_ProGT,
    'PVRCNN_SSL_CER': PVRCNN_SSL_CER,
    'PVRCNN_SSL_Consis': PVRCNN_SSL_Consis,

    'PVRCNN_Pseudo': PVRCNN_Pseudo,

}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

_base_ = './default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 12]
    }
)

OptimizationParams = dict(
    stat_npts = 20000,
    dyn_npts = 10000,
    # depth_warp_iterations = 3000,
    densify = 3,
    desicnt = 12,
    lambda_flow_loss = 0
)
from tensorflow import optimizers as opt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Activation, Concatenate, UpSampling1D,\
  Conv1DTranspose, Cropping1D

def compile_model(
    input_shape: tuple = (None,),
    optimizer: opt.Optimizer = None,
    starting_kernel_count: int = 32,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype'
):
    """
    https://arxiv.org/pdf/1505.04597.pdf -> The network does not have any fully connected layers
                and only uses the valid part of each convolution, i.e., the segmentation map only
                contains the pixels, for which the full context is available in the input image.
                This strategy allows the seamless segmentation of arbitrarily large images by an
                overlap-tile strategy (see Figure 2). To predict the pixels in the border region
                of the image, the missing context is extrapolated by mirroring the input image.
                This tiling strategy is important to apply the network to large images, since
                otherwise the resolution would be limited by the GPU memory.

    Args:
        input_shape (tuple, optional): _description_. Defaults to (None,).
        optimizer (opt.Optimizer, optional): _description_. Defaults to None.
        starting_kernel_count (int, optional): _description_. Defaults to 32.
        loss_func (str, optional): _description_. Defaults to None.
        eval_metrics (list, optional): _description_. Defaults to [].
        model_id (str, optional): _description_. Defaults to 'prototype'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    model_id = f'unet_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=3e-4)
    if loss_func is None:
        raise ValueError('No loss function specified')

    model_input = Input(shape=input_shape)
    ################################################################################
    contraction_1st_block = Conv1D(
        filters=starting_kernel_count,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(model_input)
    contraction_1st_block = Activation('relu')(contraction_1st_block)
    contraction_1st_block = Conv1D(
        filters=starting_kernel_count,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_1st_block)
    contraction_1st_block = Activation('relu')(contraction_1st_block)
    ################################################################################
    contraction_2nd_block = MaxPool1D(
        pool_size=2,
        strides=2
    )(contraction_1st_block)
    contraction_2nd_block = Conv1D(
        filters=starting_kernel_count * 2,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_2nd_block)
    contraction_2nd_block = Activation('relu')(contraction_2nd_block)
    contraction_2nd_block = Conv1D(
        filters=starting_kernel_count * 2,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_2nd_block)
    contraction_2nd_block = Activation('relu')(contraction_2nd_block)
    ################################################################################
    contraction_3rd_block = MaxPool1D(
        pool_size=2,
        strides=2
    )(contraction_2nd_block)
    contraction_3rd_block = Conv1D(
        filters=starting_kernel_count * 4,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_3rd_block)
    contraction_3rd_block = Activation('relu')(contraction_3rd_block)
    contraction_3rd_block = Conv1D(
        filters=starting_kernel_count * 4,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_3rd_block)
    contraction_3rd_block = Activation('relu')(contraction_3rd_block)
    ################################################################################
    contraction_4th_block = MaxPool1D(
        pool_size=2,
        strides=2
    )(contraction_3rd_block)
    contraction_4th_block = Conv1D(
        filters=starting_kernel_count * 8,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_4th_block)
    contraction_4th_block = Activation('relu')(contraction_4th_block)
    contraction_4th_block = Conv1D(
        filters=starting_kernel_count * 8,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_4th_block)
    contraction_4th_block = Activation('relu')(contraction_4th_block)
    ################################################################################
    contraction_5th_block = MaxPool1D(
        pool_size=2,
        strides=2
    )(contraction_4th_block)
    contraction_5th_block = Conv1D(
        filters=starting_kernel_count * 16,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_5th_block)
    contraction_5th_block = Activation('relu')(contraction_5th_block)
    contraction_5th_block = Conv1D(
        filters=starting_kernel_count * 16,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(contraction_5th_block)
    contraction_5th_block = Activation('relu')(contraction_5th_block)
    ################################################################################
    ################################################################################
    ################################################################################
    # expansion_1st_block = UpSampling1D(size=2)(contraction_5th_block)
    expansion_1st_block = Conv1DTranspose(
        filters=starting_kernel_count * 8,
        kernel_size=2,
        strides=2,
        padding='valid',
        output_padding=None,
        dilation_rate=1,
    )(contraction_5th_block)
    contraction_4th_block = Cropping1D(cropping=4)(contraction_4th_block)
    expansion_1st_block = Concatenate(axis=2)([contraction_4th_block,expansion_1st_block])
    expansion_1st_block = Conv1D(
        filters=starting_kernel_count * 8,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_1st_block)
    expansion_1st_block = Activation('relu')(expansion_1st_block)
    expansion_1st_block = Conv1D(
        filters=starting_kernel_count * 8,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_1st_block)
    expansion_1st_block = Activation('relu')(expansion_1st_block)
    ################################################################################
    # expansion_2nd_block = UpSampling1D(size=2)(expansion_1st_block)
    expansion_2nd_block = Conv1DTranspose(
        filters=starting_kernel_count * 4,
        kernel_size=2,
        strides=2,
        padding='valid',
        output_padding=None,
        dilation_rate=1,
    )(expansion_1st_block)
    contraction_3rd_block = Cropping1D(cropping=16)(contraction_3rd_block)
    expansion_2nd_block = Concatenate(axis=2)([contraction_3rd_block,expansion_2nd_block])
    expansion_2nd_block = Conv1D(
        filters=starting_kernel_count * 4,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_2nd_block)
    expansion_2nd_block = Activation('relu')(expansion_2nd_block)
    expansion_2nd_block = Conv1D(
        filters=starting_kernel_count * 4,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_2nd_block)
    expansion_2nd_block = Activation('relu')(expansion_2nd_block)
    ################################################################################
    # expansion_3rd_block = UpSampling1D(size=2)(expansion_2nd_block)
    expansion_3rd_block = Conv1DTranspose(
        filters=starting_kernel_count * 2,
        kernel_size=2,
        strides=2,
        padding='valid',
        output_padding=None,
        dilation_rate=1,
    )(expansion_2nd_block)
    contraction_2nd_block = Cropping1D(cropping=40)(contraction_2nd_block)
    expansion_3rd_block = Concatenate(axis=2)([contraction_2nd_block,expansion_3rd_block])
    expansion_3rd_block = Conv1D(
        filters=starting_kernel_count * 2,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_3rd_block)
    expansion_3rd_block = Activation('relu')(expansion_3rd_block)
    expansion_3rd_block = Conv1D(
        filters=starting_kernel_count * 2,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_3rd_block)
    expansion_3rd_block = Activation('relu')(expansion_3rd_block)
    ################################################################################
    # expansion_4th_block = UpSampling1D(size=2)(expansion_3rd_block)
    expansion_4th_block = Conv1DTranspose(
        filters=starting_kernel_count,
        kernel_size=2,
        strides=2,
        padding='valid',
        output_padding=None,
        dilation_rate=1,
    )(expansion_3rd_block)
    contraction_1st_block = Cropping1D(cropping=88)(contraction_1st_block)
    expansion_4th_block = Concatenate(axis=2)([contraction_1st_block,expansion_4th_block])
    expansion_4th_block = Conv1D(
        filters=starting_kernel_count,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_4th_block)
    expansion_4th_block = Activation('relu')(expansion_4th_block)
    expansion_4th_block = Conv1D(
        filters=starting_kernel_count,
        kernel_size=3,
        strides=1,
        padding='valid',
    )(expansion_4th_block)
    expansion_4th_block = Activation('relu')(expansion_4th_block)
    ################################################################################
    output = Conv1D(
        filters=2,
        kernel_size=1,
    )(expansion_4th_block)

    model = Model(
        model_input,
        output,
        name=model_id
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=eval_metrics
    )

    return model
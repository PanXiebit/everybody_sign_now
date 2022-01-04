# everybody_sign_now
An unofficial implementation of ["Can Everybody Sign Now? Exploring Sign Language Video Generation from 2D Poses"](https://arxiv.org/pdf/2012.10941.pdf).


## Dataset Preparation
- how2sign: https://how2sign.github.io/#download

    ```bash
    - Data
        -how2sign
            - train
                - openpose_output
                    - json
                    - video
                - videos
            - val
                - openpose_output
                    - json
                    - video
                - videos
        how2sign_train.csv
    how2sign_val.csv
    ```

This source code is inspired by both [Everybody Dance Now](https://carolineec.github.io/everybody_dance_now/).


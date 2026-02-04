# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D


def build_sam3D_vit_h(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam3D = build_sam3D_vit_h


def build_sam3D_vit_l(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam3D_vit_b(checkpoint=None):
    return _build_sam3D(
        # encoder_embed_dim=768,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_sam3D_vit_b_ori(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry3D = {
    "default": build_sam3D_vit_h,
    "vit_h": build_sam3D_vit_h,
    "vit_l": build_sam3D_vit_l,
    "vit_b": build_sam3D_vit_b,
    "vit_b_ori": build_sam3D_vit_b_ori,
}


def _build_sam3D(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size,
                                  image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_sam3D_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    """
    Fonction interne pour construire un modèle SAM-Med3D avec configuration originale.

    Cette version utilise une taille d'image réduite (128x128x128) par rapport à la
    version standard (256x256x256). Elle est conçue pour :
    - Compatibilité avec les poids pré-entraînés originaux
    - Utilisation sur des GPU avec mémoire limitée
    - Traitement plus rapide des volumes médicaux

    La réduction de la taille d'image de 256 à 128 divise la mémoire requise par ~8
    (128³ vs 256³) et accélère significativement le traitement.

    Arguments:
        encoder_embed_dim (int): Dimension des embeddings de l'encodeur (typiquement 768).
        encoder_depth (int): Nombre de blocs transformer dans l'encodeur.
        encoder_num_heads (int): Nombre de têtes d'attention multi-têtes.
        encoder_global_attn_indexes (list): Indices des couches avec attention globale.
        checkpoint (str, optional): Chemin vers les poids pré-entraînés.

    Returns:
        Sam3D: Modèle SAM-Med3D avec configuration originale (image_size=128).
    """
    # Dimension d'embedding fixe pour les invites et le décodeur
    prompt_embed_dim = 384

    # Taille d'image RÉDUITE pour la version originale (128 au lieu de 256)
    # Volumes 3D de 128x128x128 voxels - plus rapide mais moins de détails
    image_size = 128

    # Taille des patches pour le Vision Transformer (identique)
    vit_patch_size = 16

    # Calculer la taille spatiale des embeddings d'image
    # 128 // 16 = 8, donc les embeddings seront de forme 8x8x8
    # (Moitié de la résolution par rapport à la version standard)
    image_embedding_size = image_size // vit_patch_size

    # Construire le modèle SAM-Med3D avec la configuration originale
    sam = Sam3D(
        # Encodeur d'image 3D : architecture identique mais traite des volumes plus petits
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,  # 128 au lieu de 256
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        # Encodeur d'invites 3D : adapté à la résolution réduite
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            # Taille d'embedding réduite (8x8x8 au lieu de 16x16x16)
            image_embedding_size=(image_embedding_size, image_embedding_size,
                                  image_embedding_size),
            # Taille d'image d'entrée réduite
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        # Décodeur de masques 3D : configuration identique
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # Normalisation identique à la version standard
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Mettre le modèle en mode évaluation
    sam.eval()

    # Charger les poids pré-entraînés si disponibles
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)

    return sam

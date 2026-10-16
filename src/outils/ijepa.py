import random
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class JEPAMaskingConfig:
    """
    Configuration du masquage selon l'article I-JEPA.
    Paramètres modifiables à l'instanciation.
    """
    
    def __init__(self, 
                 img_size=224, 
                 patch_size=16,
                 target_scale_min=0.15,
                 target_scale_max=0.20,
                 target_aspect_min=0.75,
                 target_aspect_max=1.5,
                 num_targets=4,
                 context_scale_min=0.85,
                 context_scale_max=1.0,
                 context_aspect=1.0):
        """
        Args:
            img_size: taille de l'image en pixels (H=W)
            patch_size: taille de chaque patch en pixels
            target_scale_min: échelle minimale des blocs cibles
            target_scale_max: échelle maximale des blocs cibles
            target_aspect_min: ratio d'aspect minimal des cibles
            target_aspect_max: ratio d'aspect maximal des cibles
            num_targets: nombre de blocs cibles (M)
            context_scale_min: échelle minimale du bloc contexte
            context_scale_max: échelle maximale du bloc contexte
            context_aspect: ratio d'aspect du contexte (1.0 = carré)
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Paramètres des cibles (target blocks)
        self.target_scale_min = target_scale_min
        self.target_scale_max = target_scale_max
        self.target_aspect_min = target_aspect_min
        self.target_aspect_max = target_aspect_max
        self.num_targets = num_targets
        
        # Paramètres du contexte (context block)
        self.context_scale_min = context_scale_min
        self.context_scale_max = context_scale_max
        self.context_aspect = context_aspect

    @classmethod
    def standard(cls, img_size=64, patch_size=8):
        """Configuration conforme à l'article I-JEPA"""
        return cls(img_size=img_size, patch_size=patch_size)

    @classmethod
    def harder(cls, img_size=64, patch_size=8):
        """Configuration plus difficile pour lutter contre l'overfitting"""
        return cls(
            img_size=img_size,
            patch_size=patch_size,
            target_scale_min=0.20,
            target_scale_max=0.35,
            num_targets=6,
            context_scale_min=0.70,
            context_scale_max=0.90
        )

    @classmethod
    def very_hard(cls, img_size=64, patch_size=8):
        """Configuration très difficile"""
        return cls(
            img_size=img_size,
            patch_size=patch_size,
            target_scale_min=0.30,
            target_scale_max=0.45,
            num_targets=8,
            context_scale_min=0.50,
            context_scale_max=0.70
        )
class JEPAMaskGenerator:
    """
    Générateur de masques pour I-JEPA.
    Produit les indices des patches pour le contexte et les cibles.
    
    Référence: Section 3 - Method, "Masking strategy"
    """
    
    def __init__(self, config):
        self.config = config
        
    def _random_block(self, scale_min, scale_max, aspect_min=None, aspect_max=None, aspect_fixed=None):
       
        grid = self.config.grid_size
        
        # Échelle en nombre de patches
        scale = random.uniform(scale_min, scale_max)
        area = scale * scale * grid * grid  # surface en nombre de patches
        
        # Ratio d'aspect
        if aspect_fixed is not None:
            aspect = aspect_fixed
        else:
            aspect = random.uniform(aspect_min, aspect_max)
        
        # Dimensions en patches
        w_patches = int(round(math.sqrt(area * aspect)))
        h_patches = int(round(math.sqrt(area / aspect)))
        
        # Ajuster pour rester dans les limites
        w_patches = min(w_patches, grid - 1)
        h_patches = min(h_patches, grid - 1)
        
        if w_patches < 1:
            w_patches = 1
        if h_patches < 1:
            h_patches = 1
        
        # Position aléatoire
        x = random.randint(0, grid - w_patches)
        y = random.randint(0, grid - h_patches)
        
        return x, y, w_patches, h_patches
    
    def _block_to_indices(self, x, y, w, h):
        
        grid = self.config.grid_size
        indices = []
        for row in range(y, y + h):
            for col in range(x, x + w):
                idx = row * grid + col
                indices.append(idx)
        return indices
    
    def generate_masks(self, batch_size):
        
        context_masks = []
        target_masks_list = []
        
        for _ in range(batch_size):
            # 1. Générer les blocs cibles (M=4)
            target_blocks = []
            target_blocks_coords = []
            
            for _ in range(self.config.num_targets):
                x, y, w, h = self._random_block(
                    self.config.target_scale_min,
                    self.config.target_scale_max,
                    self.config.target_aspect_min,
                    self.config.target_aspect_max
                )
                target_blocks.append(self._block_to_indices(x, y, w, h))
                target_blocks_coords.append((x, y, w, h))
            
            # 2. Générer le bloc contexte
            ctx_x, ctx_y, ctx_w, ctx_h = self._random_block(
                self.config.context_scale_min,
                self.config.context_scale_max,
                aspect_fixed=self.config.context_aspect
            )
            context_idx = set(self._block_to_indices(ctx_x, ctx_y, ctx_w, ctx_h))
            
            # 3. Supprimer les chevauchements avec les cibles
            #    (comme décrit dans l'article: "remove any overlapping regions from the context block")
            for target_block in target_blocks:
                context_idx = context_idx - set(target_block)
            
            # Sauvegarder
            context_masks.append(list(context_idx))
            target_masks_list.append(target_blocks)
        
        return context_masks, target_masks_list
    
    def visualize_masks(self, image, context_mask, target_masks):
       
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        h, w = image.shape[:2]
        
        # Créer une copie
        vis = image.copy().astype(np.uint8)
        
        # Couleurs pour les cibles (RGB)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        # Dessiner les cibles
        for i, target_mask in enumerate(target_masks):
            color = colors[i % len(colors)]
            for idx in target_mask:
                row = idx // grid
                col = idx % grid
                y1 = row * patch_size
                x1 = col * patch_size
                y2 = y1 + patch_size
                x2 = x1 + patch_size
                # Mélanger l'image originale avec la couleur
                vis[y1:y2, x1:x2] = vis[y1:y2, x1:x2] * 0.4 + np.array(color) * 0.6
        
        # Dessiner le contexte (en cyan)
        for idx in context_mask:
            row = idx // grid
            col = idx % grid
            y1 = row * patch_size
            x1 = col * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size
            vis[y1:y2, x1:x2] = vis[y1:y2, x1:x2] * 0.4 + np.array([0, 255, 255]) * 0.6
        
        return vis

class IJEPADataGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données pour I-JEPA avec masquage.
    Retourne ((images_masquees, target_data), labels)
    """
    
    def __init__(self, images, labels=None, batch_size=32, img_size=64, patch_size=8, shuffle=True, **kwargs):

        super().__init__(**kwargs)
        self.images = images.astype(np.float32)
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        
        self.config = JEPAMaskingConfig.standard(img_size=64, patch_size=8)
        self.mask_generator = JEPAMaskGenerator(self.config)
        
        self.indices = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def _apply_mask_to_image(self, image, context_mask):
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        masked_image = np.zeros_like(image, dtype=np.float32)
        
        for idx in context_mask:
            row = idx // grid
            col = idx % grid
            y1 = row * patch_size
            x1 = col * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size
            masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        
        return masked_image
    
    def _pad_target_masks(self, target_masks_list):
        """
        Pad les target_masks pour qu'ils aient tous la même taille.
        """
        batch_size = len(target_masks_list)
        if batch_size == 0:
            return np.array([]), np.array([])
        
        num_targets = len(target_masks_list[0])
        
        #  Calcul correct de max_patches
        max_patches = 0
        for masks in target_masks_list:
            for m in masks:
                max_patches = max(max_patches, len(m))
        
        #  Si max_patches est trop petit, au moins 4
        max_patches = max(max_patches, 4)
        
        padded_masks = np.zeros((batch_size, num_targets, max_patches), dtype=np.int32)
        mask_lengths = np.zeros((batch_size, num_targets), dtype=np.int32)
        
        for i, masks in enumerate(target_masks_list):
            for j, m in enumerate(masks):
                length = len(m)
                mask_lengths[i, j] = length
                if length > 0:
                    padded_masks[i, j, :length] = m
        
        return padded_masks, mask_lengths
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_images = batch_images / 255.0
        
        # Générer les masques
        context_masks, target_masks_list = self.mask_generator.generate_masks(len(batch_images))
        
        # Appliquer les masques
        context_images = []
        for i, img in enumerate(batch_images):
            masked = self._apply_mask_to_image(img, context_masks[i])
            context_images.append(masked)
        context_images = np.array(context_images, dtype=np.float32)
        
        # Pad les target_masks
        padded_target_masks, mask_lengths = self._pad_target_masks(target_masks_list)
        
        # Convertir en tenseurs
        padded_target_masks = tf.constant(padded_target_masks, dtype=tf.int32)
        mask_lengths = tf.constant(mask_lengths, dtype=tf.int32)
        target_data = (padded_target_masks, mask_lengths)
        
        if self.labels is not None:
            batch_labels = self.labels[batch_indices]
            batch_labels = tf.constant(batch_labels, dtype=tf.int32)
            return (context_images, target_data), batch_labels
        else:
            dummy_labels = tf.zeros(len(batch_indices), dtype=tf.int32)
            return (context_images, target_data), dummy_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def visualize_batch(self, batch_idx=0, save_path=None):
        
        
        # Récupérer le batch
        (context_images, target_data), labels = self[batch_idx]
        padded_target_masks, mask_lengths = target_data
        
        # Récupérer les images originales
        batch_indices = self.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        original_images = self.images[batch_indices].astype(np.uint8)
        
        #  Vérifier le type et convertir si nécessaire
        if hasattr(context_images, 'numpy'):
            context_images = context_images.numpy()
        if hasattr(padded_target_masks, 'numpy'):
            padded_target_masks = padded_target_masks.numpy()
        if hasattr(mask_lengths, 'numpy'):
            mask_lengths = mask_lengths.numpy()
        
        # Reconstruire les target_masks
        target_masks_list = []
        for i in range(padded_target_masks.shape[0]):
            masks = []
            for j in range(padded_target_masks.shape[1]):
                length = mask_lengths[i, j] if mask_lengths.ndim > 1 else mask_lengths[j]
                if length > 0:
                    masks.append(padded_target_masks[i, j, :length].tolist())
                else:
                    masks.append([])
            target_masks_list.append(masks)
        
        # Générer les masques de contexte pour la visualisation
        context_masks, _ = self.mask_generator.generate_masks(len(original_images))
        
        n_images = min(2, len(context_images))
        fig, axes = plt.subplots(n_images, 4, figsize=(10, 2 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            # 1. Image originale
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title(f"Originale {i+1}")
            axes[i, 0].axis('off')
            
            # 2. Image masquée (seul le contexte)
            axes[i, 1].imshow(context_images[i])
            axes[i, 1].set_title(f"Contexte masqué")
            axes[i, 1].axis('off')
            
            # 3. Visualisation des cibles
            vis_targets = self.mask_generator.visualize_masks(
                original_images[i],
                [],  # pas de contexte pour cette visualisation
                target_masks_list[i]
            )
            axes[i, 2].imshow(vis_targets)
            axes[i, 2].set_title(f"Cibles ({self.config.num_targets} blocs)")
            axes[i, 2].axis('off')
            
            # 4. Visualisation complète (contexte + cibles)
            vis_full = self.mask_generator.visualize_masks(
                original_images[i],
                context_masks[i],
                target_masks_list[i]
            )
            axes[i, 3].imshow(vis_full)
            axes[i, 3].set_title(f"Contexte + Cibles")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
            
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
        plt.close(fig)





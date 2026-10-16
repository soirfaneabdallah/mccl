import os
import copy
import random
import io
import zipfile
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

class Load_data:
    """
    Classe pour charger, préparer et gérer des ensembles d'images pour l'entraînement de modèles ML/DL.
    Parcourt récursivement tous les sous-dossiers pour trouver les images.
    """

    def __init__(self, 
                 root_path: str = None, 
                 path_list: list[str] = None, 
                 extension: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff'),
                 image_shape: tuple[int, int, int] = (240, 240, 3)) -> None:
        """
        Initialise la classe.
        
        :param root_path: dossier racine à parcourir récursivement
        :param path_list: liste alternative de dossiers spécifiques
        :param extension: extensions des fichiers supportés
        :param image_shape: tuple (H, W, C) -> hauteur, largeur, canaux (1 = grayscale, 3 = RGB)
        """
        self.root_path_ = root_path
        self.path_list_ = path_list
        self.extension_ = extension
        self.name_label_ = []
        self.data_ = None
        self.data_label_ = None
        self.original_data_ = None
        self.image_shape_ = image_shape  # (H, W, C)
        self.class_mapping_ = {}  # mapping entre classes et labels encodés

    # ---------------------- Parcours récursif des dossiers ----------------------
    def _get_image_folders(self, root_path: str) -> list:
        """
        Parcourt récursivement tous les sous-dossiers pour trouver ceux qui contiennent des images.
        Retourne une liste de dossiers qui contiennent directement des images.
        """
        image_folders = []
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Vérifier si ce dossier contient au moins une image
            has_images = any(f.lower().endswith(self.extension_) for f in filenames)
            
            if has_images:
                # Le dossier contient des images directement
                image_folders.append(dirpath)
            else:
                # Vérifier si les sous-dossiers contiennent des images
                for dirname in dirnames:
                    subdir_path = os.path.join(dirpath, dirname)
                    if any(f.lower().endswith(self.extension_) for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))):
                        image_folders.append(subdir_path)
        
        # Supprimer les doublons (au cas où un dossier parent serait aussi compté)
        image_folders = list(set(image_folders))
        
        return image_folders

    def _get_label_from_path(self, image_path: str, root_path: str) -> str:
        """
        Extrait le label à partir du chemin de l'image.
        Le label est le nom du dossier parent immédiat de l'image.
        """
        # Obtenir le dossier contenant l'image
        image_dir = os.path.dirname(image_path)
        # Le label est le nom de ce dossier
        label = os.path.basename(image_dir)
        return label

    # ---------------------- Chargement ----------------------
    def load(self, recursive: bool = True) -> None:
        """
        Charge et prétraite les images.
        
        :param recursive: si True, parcourt récursivement tous les sous-dossiers.
                         si False, charge seulement les dossiers explicitement listés.
        """
        image_data, labels = [], []
        H, W, C = self.image_shape_
        
        # Déterminer la liste des dossiers à parcourir
        folders_to_scan = []
        
        if self.root_path_ and recursive:
            # Parcours récursif du dossier racine
            folders_to_scan = self._get_image_folders(self.root_path_)
            print(f"Trouvé {len(folders_to_scan)} dossiers contenant des images")
        elif self.path_list_:
            # Utilisation de la liste explicite
            folders_to_scan = self.path_list_
        
        if not folders_to_scan:
            print("Aucun dossier à scanner. Vérifiez root_path_ ou path_list_.")
            return
        
        # Chargement des images
        for folder in tqdm(folders_to_scan, desc="Chargement des dossiers"):
            # Le label est le nom du dossier courant (celui qui contient directement les images)
            label = os.path.basename(os.path.normpath(folder))
            
            if label not in self.name_label_:
                self.name_label_.append(label)
            
            for fichier in os.listdir(folder):
                if fichier.lower().endswith(self.extension_):
                    try:
                        path_image = os.path.join(folder, fichier)
                        img = Image.open(path_image)
                        
                        # Convertir selon le nombre de canaux
                        if C == 1:
                            img = img.convert("L")  # grayscale
                        elif C == 3:
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Le nombre de canaux doit être 1 (gris) ou 3 (RGB).")
                        
                        # Redimensionnement
                        img = img.resize((W, H))
                        img = np.array(img)
                        
                        # Ajout d'une dimension pour grayscale
                        if C == 1:
                            img = np.expand_dims(img, axis=-1)
                        
                        image_data.append(img)
                        labels.append(label)
                        
                    except (OSError, ValueError, Exception) as e:
                        print(f"Erreur lors du chargement de {fichier}: {e}")
                        continue
        
        print(f"Images chargées : {len(image_data)}")
        
        # Création du DataFrame
        self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        self.data_label_ = pd.DataFrame({'Label': labels})
        self.copy()
        
        # Affichage du nombre de classes trouvées
        unique_labels = self.data_['Label'].unique()
        print(f"Classes trouvées : {len(unique_labels)}")
        for lbl in unique_labels:
            count = len(self.data_[self.data_['Label'] == lbl])
            print(f"  - {lbl}: {count} images")

    def load_from_multiple_roots(self, root_paths: list[str], recursive: bool = True) -> None:
        """
        Charge des images depuis plusieurs dossiers racines.
        
        :param root_paths: liste des dossiers racines à parcourir
        :param recursive: si True, parcourt récursivement les sous-dossiers
        """
        all_folders = []
        
        for root in root_paths:
            if recursive:
                folders = self._get_image_folders(root)
                all_folders.extend(folders)
            else:
                all_folders.append(root)
        
        # Supprimer les doublons
        all_folders = list(set(all_folders))
        
        self.path_list_ = all_folders
        self.root_path_ = None
        self.load(recursive=False)

    # ---------------------- Mélange ----------------------
    def shuffle(self) -> None:
        """ Mélange aléatoirement les images et labels. """
        if self.data_ is not None:
            self.data_ = self.data_.sample(frac=1, random_state=42).reset_index(drop=True)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})

    # ---------------------- Sauvegarde état ----------------------
    def copy(self) -> None:
        """ Sauvegarde une copie des données originales. """
        if self.data_ is not None:
            self.original_data_ = copy.deepcopy(self.data_)

    def restore_data(self) -> None:
        """ Restaure les données originales. """
        if self.original_data_ is not None:
            self.data_ = copy.deepcopy(self.original_data_)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})

    # ---------------------- Ajout de nouvelles données ----------------------
    def add_data(self, folder: str, recursive: bool = True) -> None:
        """
        Ajoute des images depuis un nouveau dossier ou une nouvelle racine.
        
        :param folder: chemin vers le dossier ou la racine
        :param recursive: si True, parcourt récursivement les sous-dossiers
        """
        H, W, C = self.image_shape_
        
        # Déterminer les dossiers à scanner
        if recursive and os.path.isdir(folder):
            folders_to_scan = self._get_image_folders(folder)
        else:
            folders_to_scan = [folder]
        
        image_data, labels = [], []
        
        for scan_folder in folders_to_scan:
            label = os.path.basename(os.path.normpath(scan_folder))
            
            # Si la classe est nouvelle, on l'ajoute
            if label not in self.name_label_:
                self.name_label_.append(label)
            
            for fichier in tqdm(os.listdir(scan_folder), desc=f"Ajout: {label}"):
                if fichier.lower().endswith(self.extension_):
                    try:
                        path_image = os.path.join(scan_folder, fichier)
                        img = Image.open(path_image)
                        
                        # Conversion selon shape
                        if C == 1:
                            img = img.convert("L")
                        elif C == 3:
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Le nombre de canaux doit être 1 (gris) ou 3 (RGB).")
                        
                        # Redimensionnement
                        img = img.resize((W, H))
                        img = np.array(img)
                        
                        if C == 1:
                            img = np.expand_dims(img, axis=-1)
                        
                        image_data.append(img)
                        labels.append(label)
                        
                    except (OSError, ValueError, Exception) as e:
                        print(f"Erreur: {e}")
                        continue
        
        # Fusion avec le dataset existant
        if self.data_ is None:
            self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        else:
            new_df = pd.DataFrame({'Image': image_data, 'Label': labels})
            self.data_ = pd.concat([self.data_, new_df], ignore_index=True)
        
        # Mise à jour des labels
        self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
        self.copy()
        
        print(f"Ajouté : {len(image_data)} images")
        print(f"Total : {len(self.data_)} images, {len(self.name_label_)} classes")

    # ---------------------- Visualisation ----------------------
    def plot(self, view_code: bool = False, name_fig: str = "fig", register: bool = False, n_samples: int = 20) -> None:
        """
        Affiche un échantillon aléatoire d'images avec leurs labels.
        
        :param view_code: afficher les codes numériques plutôt que les noms
        :param name_fig: nom du fichier pour la sauvegarde
        :param register: sauvegarder la figure
        :param n_samples: nombre d'images à afficher (doit être un multiple de 4 ou 5)
        """
        if self.data_ is None:
            print("Aucune donnée chargée.")
            return
        
        n_samples = min(n_samples, len(self.data_))
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        indices = np.random.choice(len(self.data_), n_samples, replace=False)
        
        title_source = self.data_["Label"] if not view_code else self.data_label_["Label"]
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.data_["Image"][idx])
            axes[i].set_title(str(title_source[idx]))
            axes[i].axis("off")
        
        # Cacher les axes inutilisés
        for i in range(len(indices), len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        if register:
            plt.savefig(f"{name_fig}.png", dpi=300, bbox_inches='tight')
        plt.show()

    # ---------------------- Traitement des labels ----------------------
    def encodage(self) -> None:
        """
        Encode les labels en entiers.
        """
        if self.data_ is None:
            return
        
        if isinstance(self.data_["Label"].iloc[0], str):
            self.class_mapping_ = {name: i for i, name in enumerate(self.name_label_)}
            self.data_["Label"] = self.data_["Label"].map(self.class_mapping_)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
            print(f"Encodage effectué. Mapping: {self.class_mapping_}")

    def decode_labels(self, encoded_labels: list) -> list:
        """
        Décode des labels encodés en noms de classes.
        
        :param encoded_labels: liste des labels encodés
        :return: liste des noms de classes
        """
        reverse_mapping = {v: k for k, v in self.class_mapping_.items()}
        return [reverse_mapping.get(label, "unknown") for label in encoded_labels]

    # ---------------------- Compression ----------------------
    def compress(self, image: np.ndarray, test: bool = True, k: int = 100, threshold_kb: float = 200.0) -> np.ndarray:
        """
        Compresse une image via décomposition SVD si elle dépasse un seuil en Ko.
        """
        image_size_kb = image.nbytes / 1024
        if image_size_kb > threshold_kb:
            compressed_channels = []
            H, W = image.shape[:2]
            for i in range(image.shape[2] if image.ndim == 3 else 1):
                channel = image[:, :, i] if image.ndim == 3 else image
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                Sk = np.diag(S[:k])
                compressed = np.dot(U[:, :k], np.dot(Sk, Vt[:k, :]))
                compressed_channels.append(compressed)
            
            if len(compressed_channels) > 1:
                compressed_img = np.stack(compressed_channels, axis=2)
            else:
                compressed_img = compressed_channels[0]
            
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
            resized_img = Image.fromarray(compressed_img.astype(np.uint8)).resize((W, H))
            
            if test:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(resized_img)
                ax[0].set_title("Image compressée")
                ax[1].imshow(image.astype(np.uint8))
                ax[1].set_title("Image originale")
                plt.show()
            
            return np.array(resized_img) / 255.0
        
        return np.array(image) / 255.0

    def compress_data(self, threshold_kb: float = 200.0) -> None:
        """
        Compresse toutes les images de l'ensemble de données.
        """
        if self.data_ is None:
            return
        
        compressed_images = []
        for img in tqdm(self.data_["Image"], desc="Compression"):
            compressed_images.append(self.compress(img, test=False, threshold_kb=threshold_kb))
        
        self.data_["Image"] = compressed_images

    # ---------------------- Sauvegarde / Chargement ----------------------
    def save(self, zip_path: str = "dataset.zip", internal_name: str = "loader.pkl") -> None:
        """
        Sauvegarde l'objet dans une archive zip.
        """
        backup = self.data_
        self.data_ = None
        buffer = io.BytesIO()
        pickle.dump(self, buffer)
        self.data_ = backup
        buffer.seek(0)
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(internal_name, buffer.read())
        
        print(f"Dataset sauvegardé dans {zip_path}")

    @staticmethod
    def load_object(zip_path: str = "dataset.zip", 
                    internal_name: str = "loader.pkl",
                    selected_classes: list[str] = None,
                    min_images_per_class: int = None,
                    max_images_per_class: int = None) -> "Load_data":
        """
        Charge un objet Load_data sauvegardé dans une archive zip avec possibilité de filtrer les classes.
        
        :param zip_path: chemin vers l'archive zip
        :param internal_name: nom interne du fichier dans l'archive
        :param selected_classes: liste des classes à charger (ex: ['plastique', 'cannette', 'verre'])
        :param min_images_per_class: nombre minimum d'images par classe (ex: 100)
        :param max_images_per_class: nombre maximum d'images par classe (ex: 500)
        :return: objet Load_data filtré
        """
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_name) as f:
                obj = pickle.load(f)
                
                # Restaurer les données originales si disponibles
                if obj.original_data_ is not None:
                    obj.data_ = copy.deepcopy(obj.original_data_)
                
                # Recréer data_label_ si nécessaire
                if obj.data_ is not None and obj.data_label_ is None:
                    obj.data_label_ = pd.DataFrame({'Label': obj.data_["Label"].tolist()})
                
                # S'assurer que name_label_ existe
                if not hasattr(obj, "name_label_"):
                    obj.name_label_ = []
                
                # Appliquer les filtres si nécessaire
                if selected_classes is not None or min_images_per_class is not None or max_images_per_class is not None:
                    obj._filter_classes(selected_classes, min_images_per_class, max_images_per_class)
                
                return obj

    def _filter_classes(self, 
                        selected_classes: list[str] = None, 
                        min_images_per_class: int = None, 
                        max_images_per_class: int = None) -> None:
        """
        Filtre les classes du dataset selon différents critères.
        
        :param selected_classes: liste des classes à conserver
        :param min_images_per_class: nombre minimum d'images par classe
        :param max_images_per_class: nombre maximum d'images par classe
        """
        if self.data_ is None:
            print("Aucune donnée à filtrer.")
            return
        
        original_count = len(self.data_)
        original_classes = len(self.name_label_)
        
        # Compter le nombre d'images par classe
        class_counts = self.data_['Label'].value_counts()
        
        # Déterminer les classes à conserver
        classes_to_keep = set()
        
        # Cas 1: Filtrage par liste de classes spécifiques
        if selected_classes is not None:
            selected_set = set(selected_classes)
            classes_to_keep.update(selected_set)
        
        # Cas 2: Filtrage par nombre minimum d'images
        if min_images_per_class is not None:
            for label, count in class_counts.items():
                if count >= min_images_per_class:
                    classes_to_keep.add(label)
        
        # Cas 3: Filtrage par nombre maximum d'images
        if max_images_per_class is not None:
            temp_classes = set()
            for label, count in class_counts.items():
                if count <= max_images_per_class:
                    temp_classes.add(label)
            
            if classes_to_keep:
                classes_to_keep = classes_to_keep.intersection(temp_classes)
            else:
                classes_to_keep = temp_classes
        
        # Si aucun critère n'a été satisfait, on garde tout
        if not classes_to_keep:
            print("Aucun critère de filtrage spécifié ou aucune classe ne correspond. Conservation de toutes les classes.")
            return
        
        # Convertir les labels en string si nécessaire
        if isinstance(self.data_["Label"].iloc[0], int) and self.class_mapping_:
            # Si les labels sont encodés, on doit décoder pour la comparaison
            reverse_mapping = {v: k for k, v in self.class_mapping_.items()}
            classes_to_keep_encoded = {k for k, v in self.class_mapping_.items() if v in classes_to_keep}
            classes_to_keep = classes_to_keep_encoded if classes_to_keep_encoded else classes_to_keep
        
        # Filtrer les données
        mask = self.data_['Label'].isin(classes_to_keep)
        self.data_ = self.data_[mask].reset_index(drop=True)
        
        # Mettre à jour les labels
        self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
        
        # Mettre à jour name_label_
        if isinstance(self.data_["Label"].iloc[0], str):
            self.name_label_ = list(self.data_['Label'].unique())
        else:
            # Si labels encodés, on garde le mapping original mais on filtre name_label_
            unique_labels = self.data_['Label'].unique()
            if self.class_mapping_:
                reverse_mapping = {v: k for k, v in self.class_mapping_.items()}
                self.name_label_ = [reverse_mapping[label] for label in unique_labels if label in reverse_mapping]
            else:
                self.name_label_ = list(unique_labels)
        
        
        self.copy()

    # ---------------------- Split train/test ----------------------
    def create_data(self, test_size: float = 0.2, shuffle: bool = True):
        """
        Crée les ensembles d'entraînement et de test.
        
        :param test_size: proportion du jeu de test
        :param shuffle: mélanger les données avant le split
        :return: tuple ((X_train, Y_train), (X_test, Y_test))
        """
        if self.data_ is None:
            raise ValueError("Aucune donnée chargée. Appelez load() d'abord.")
        
        # S'assurer que les labels sont encodés
        if isinstance(self.data_["Label"].iloc[0], str):
            self.encodage()
        
        data_train, data_test = train_test_split(
            self.data_, test_size=test_size, random_state=42, shuffle=shuffle
        )
        
        X_train = np.array(data_train['Image'].tolist(), dtype=np.float32)
        Y_train = np.array(data_train['Label'].tolist(), dtype=np.int32)
        X_test = np.array(data_test['Image'].tolist(), dtype=np.float32)
        Y_test = np.array(data_test['Label'].tolist(), dtype=np.int32)
        
        return (X_train, Y_train), (X_test, Y_test)

    # ---------------------- Informations ----------------------
    def info(self) -> None:
        """Affiche les informations sur le dataset."""
        if self.data_ is None:
            print("Aucune donnée chargée.")
            return
        
        print("\n" + "="*50)
        print("INFORMATIONS SUR LE DATASET")
        print("="*50)
        print(f"Nombre total d'images : {len(self.data_)}")
        print(f"Nombre de classes : {len(self.name_label_)}")
        print(f"Shape des images : {self.image_shape_}")
        print("\nRépartition par classe :")
        
        label_counts = self.data_['Label'].value_counts()
        for label, count in label_counts.items():
            if isinstance(label, int):
                label_name = self.decode_labels([label])[0] if self.class_mapping_ else str(label)
                print(f"  - {label_name}: {count} images")
            else:
                print(f"  - {label}: {count} images")
        
        print("="*50 + "\n")


    def reshape(self, target_shape: tuple[int, int, int], batch_size: int = 32) -> None:
        
        """
        Redimensionne les images par batch vers target_shape.
       
        - Si l'image est plus grande → resize direct
        - Si plus petite → interpolation
        - Utilise tqdm pour suivi
        - Modifie self.data_["Image"] directement
        """
    
        if self.data_ is None:
            raise ValueError("Aucune donnée chargée.")
    
        H_target, W_target, C_target = target_shape
        total_images = len(self.data_)
    
        new_images = []
    
        for start in tqdm(range(0, total_images, batch_size), desc="Reshape en batch"):
            batch = self.data_["Image"].iloc[start:start + batch_size]
    
            for img in batch:
                img = np.array(img)
                H_init, W_init, C_init = img.shape
    
                # Vérification des canaux
                if C_init != C_target:
                    if C_target == 3:
                        img = Image.fromarray(img).convert("RGB")
                    elif C_target == 1:
                        img = Image.fromarray(img).convert("L")
                    img = np.array(img)
    
                # Resize nécessaire ?
                if (H_init != H_target) or (W_init != W_target):
    
                    pil_img = Image.fromarray(img)
    
                    # Si image plus petite → interpolation plus douce
                    if H_init < H_target or W_init < W_target:
                        resized = pil_img.resize((W_target, H_target), Image.BICUBIC)
                    else:
                        resized = pil_img.resize((W_target, H_target), Image.BILINEAR)
    
                    img = np.array(resized)
    
                    if C_target == 1:
                        img = np.expand_dims(img, axis=-1)
    
                new_images.append(img)
    
        self.data_["Image"] = new_images
        self.image_shape_ = target_shape

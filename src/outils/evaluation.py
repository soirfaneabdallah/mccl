import time
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from datetime import datetime
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def compare_models(y_true_test: np.ndarray,
                   y_pred_test_model1: np.ndarray,
                   y_pred_test_model2: np.ndarray,
                   y_true_local: np.ndarray,
                   y_pred_local_model1: np.ndarray,
                   y_pred_local_model2: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Compare les performances de deux modèles sur un jeu de test et un jeu local.
    
    Calcule les pourcentages de données correctement et incorrectement classifiées
    pour chaque modèle, puis affiche un graphe comparatif.

    Parameters
    ----------
    y_true_test : np.ndarray
        Labels réels du jeu de test.
    y_pred_test_model1 : np.ndarray
        Prédictions du modèle 1 sur le jeu de test.
    y_pred_test_model2 : np.ndarray
        Prédictions du modèle 2 sur le jeu de test.
    y_true_local : np.ndarray
        Labels réels du jeu local.
    y_pred_local_model1 : np.ndarray
        Prédictions du modèle 1 sur le jeu local.
    y_pred_local_model2 : np.ndarray
        Prédictions du modèle 2 sur le jeu local.

    Returns
    -------
    results : Dict[str, Tuple[float, float]]
        Dictionnaire contenant, pour chaque modèle et chaque dataset,
        un tuple (pourcentage correct, pourcentage incorrect).
    """

    results: Dict[str, Tuple[float, float]] = {}

    # --- Jeu de test ---
    acc_test_m1 = np.mean(y_true_test == y_pred_test_model1) * 100
    acc_test_m2 = np.mean(y_true_test == y_pred_test_model2) * 100
    results["Test_Model1"] = (acc_test_m1, 100 - acc_test_m1)
    results["Test_Model2"] = (acc_test_m2, 100 - acc_test_m2)

    # --- Données locales ---
    acc_local_m1 = np.mean(y_true_local == y_pred_local_model1) * 100
    acc_local_m2 = np.mean(y_true_local == y_pred_local_model2) * 100
    results["Local_Model1"] = (acc_local_m1, 100 - acc_local_m1)
    results["Local_Model2"] = (acc_local_m2, 100 - acc_local_m2)

    # --- Tracé comparatif ---
    labels = list(results.keys())
    correct = [val[0] for val in results.values()]
    incorrect = [val[1] for val in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, correct, width, label="Bien classé (%)")
    bars2 = ax.bar(x + width/2, incorrect, width, label="Mal classé (%)")

    ax.set_ylabel("Pourcentage (%)")
    ax.set_title("Comparaison des performances des modèles")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Affichage des valeurs sur les barres
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    return results

def evaluation(model, X_test, y_test, return_conf_mat=True, return_clss=True, labels=[]):
    # Prédictions
    predict_proba = model.predict(X_test)
    y_pred = np.argmax(predict_proba, axis=1)

    # Matrice de confusion
    conf_matrice = confusion_matrix(y_test, y_pred)

    if return_conf_mat:
        print("========================================== Confusion Matrix =====================================================")
        print(conf_matrice)
        f, ax = plt.subplots(figsize=(6, 5))
        conf_matrice_nor = conf_matrice.astype('float') / conf_matrice.sum(axis=1)[:, np.newaxis]
        sns.heatmap(conf_matrice_nor, annot=True, fmt=".2%", linewidths=.5, ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.title("Normalized Confusion Matrix")
        plt.show()

    if return_clss:
        print("========================================= Detailed Metrics =====================================================")
        # Rapport complet (inclut précision, rappel et f1-score par classe)
        report = classification_report(y_test, y_pred, target_names=labels, digits=4)
        print(report)

        # F1 macro et weighted
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        print(f"Macro F1-score (unweighted): {f1_macro:.4f}")
        print(f"Weighted F1-score: {f1_weighted:.4f}")
    return y_pred

def plot_result(historique, name_fig="fig", register_plot=False, register_history=True, save_dir="training_results"):
    """
    Affiche et sauvegarde les courbes d'entraînement pour TOUTES les métriques disponibles
    
    Args:
        historique : objet History retourné par model.fit()
        name_fig : nom du fichier pour la figure (sans extension)
        register_plot : bool, sauvegarde la figure si True
        register_history : bool, sauvegarde l'historique complet si True
        save_dir : répertoire de sauvegarde (créé automatiquement)
    
    Returns:
        dict: dictionnaire contenant toutes les métriques
    """
    
    # Créer le répertoire de sauvegarde si nécessaire
    if register_plot or register_history:
        os.makedirs(save_dir, exist_ok=True)
    
    # Récupérer TOUTES les métriques disponibles dans l'historique
    all_metrics = {}
    for key, value in historique.history.items():
        all_metrics[key] = value
    
    # Séparer les métriques d'entraînement et de validation
    train_metrics = {}
    val_metrics = {}
    
    for key, values in all_metrics.items():
        if key.startswith('val_'):
            val_metrics[key[4:]] = values  # enlève 'val_' du nom
        else:
            train_metrics[key] = values
    
    
    # Déterminer le nombre de graphiques
    n_plots = len(train_metrics)
    if n_plots == 0:
        print("Aucune métrique trouvée dans l'historique.")
        return {}
    
    # Créer une grille adaptative
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
    
    # Aplatir axes pour un accès facile
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    epochs = range(1, len(train_metrics[list(train_metrics.keys())[0]]) + 1)
    
    # Pour chaque métrique, créer un graphique
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        
        # Courbe d'entraînement
        ax.plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)
        
        # Courbe de validation si disponible
        if metric_name in val_metrics:
            ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Val {metric_name}', linewidth=2)
        
        
        if metric_name in val_metrics:
            best_val_idx = np.argmax(val_metrics[metric_name]) if 'acc' in metric_name.lower() or 'f1' in metric_name.lower() else np.argmin(val_metrics[metric_name])
            best_val_val = val_metrics[metric_name][best_val_idx]
            
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name.capitalize()} : Train vs Validation')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(len(train_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if register_plot:
        fig_path = os.path.join(save_dir, f"{name_fig}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {fig_path}")
    
    plt.show()
    
    # ============================================
    # Sauvegarde de l'historique complet
    # ============================================
    if register_history:
        # Préparer les données pour la sauvegarde
        history_dict = {
            'all_metrics': {},
            'summary': {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Sauvegarder toutes les métriques
        for key, values in all_metrics.items():
            history_dict['all_metrics'][key] = [float(x) for x in values]
            
            # Résumé
            if 'acc' in key.lower() or 'f1' in key.lower():
                best_idx = np.argmax(values)
                best_val = values[best_idx]
                final_val = values[-1]
            else:
                best_idx = np.argmin(values)
                best_val = values[best_idx]
                final_val = values[-1]
            
            history_dict['summary'][key] = {
                'best_value': float(best_val),
                'best_epoch': int(best_idx) + 1,
                'final_value': float(final_val)
            }
        
        # Sauvegarde en JSON
        json_path = os.path.join(save_dir, f"{name_fig}_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=4, ensure_ascii=False)
        print(f"Historique sauvegardé : {json_path}")
        
        # Sauvegarde en CSV
        csv_path = os.path.join(save_dir, f"{name_fig}_history.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            # En-tête
            headers = ['epoch'] + list(all_metrics.keys())
            f.write(','.join(headers) + '\n')
            
            # Données
            max_len = max(len(v) for v in all_metrics.values())
            for i in range(max_len):
                row = [str(i + 1)]
                for key in all_metrics.keys():
                    val = all_metrics[key][i] if i < len(all_metrics[key]) else ''
                    row.append(str(val))
                f.write(','.join(row) + '\n')
        print(f"CSV sauvegardé : {csv_path}")

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=(target_size,target_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict(img_path, discriminateur,classificateur, target_size=225, list_name = None):
    img_array, img= prepare_image(img_path , target_size)
    est_connue = discriminateur.predict(img_array)
    if est_connue[0][0] >=0.5:
        predict_proba = classificateur.predict(img_array)
        y_pred = np.argmax(predict_proba, axis=1)
        result = np.max(predict_proba[0]*100)
        plt.imshow(img)
        plt.title(f"{list_name[y_pred[0]]} avec une précision de {result}%")
    else:
        result = est_connue[0][0]*100
        plt.imshow(img)
        plt.title(f"Classe inconnue {result}%")
        
def concat_datasets(x_train, y_train, x_test, y_test,
                    x_train1, y_train1, x_test1, y_test1):
    """
    Concatène deux jeux de données (train/test).
    Les arrays doivent avoir des dimensions compatibles.
    """
    # Concat train
    X_train_full = np.concatenate([x_train, x_train1], axis=0)
    y_train_full = np.concatenate([y_train, y_train1], axis=0)

    # Concat test
    X_test_full = np.concatenate([x_test, x_test1], axis=0)
    y_test_full = np.concatenate([y_test, y_test1], axis=0)

    return X_train_full, y_train_full, X_test_full, y_test_full


def select_data(X_train, y_train, n, shuffle=True, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y_train)
    X_selected, y_selected = [], []

    for cls in classes:
        idx = np.where(y_train == cls)[0]
        if len(idx) < n:
            raise ValueError(f"Classe {cls} n'a que {len(idx)} échantillons, inférieur à n={n}.")
        chosen_idx = np.random.choice(idx, n, replace=False)
        X_selected.append(X_train[chosen_idx])
        y_selected.append(y_train[chosen_idx])

    X_selected = np.concatenate(X_selected, axis=0)
    y_selected = np.concatenate(y_selected, axis=0)

    if shuffle:
        perm = np.random.permutation(len(y_selected))
        X_selected, y_selected = X_selected[perm], y_selected[perm]

    return X_selected, y_selected
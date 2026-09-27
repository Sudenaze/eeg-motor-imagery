import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

# 1. Veriyi indir (ilk çalıştırmada otomatik iner, biraz sürer)
subject = 1
runs = [6, 10, 14]  # el ve ayak hayal etme görevleri
raw_fnames = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

eegbci.standardize(raw)
raw.set_montage(make_standard_montage("standard_1005"))

# 2. Filtrele (7-30 Hz aralığı motor imagery için tipik)
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# 3. Olayları (hangi anda ne hayal edildiğini) çıkar
events, event_id = mne.events_from_annotations(raw)
picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

# 4. Her olayın etrafından bir zaman penceresi (epoch) kes
epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0,
                     proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
X = epochs_train.get_data()
y = epochs.events[:, -1] - 2

# 5. CSP + LDA ile sınıflandır
clf = Pipeline([("CSP", CSP(n_components=4, log=True)),
                ("LDA", LinearDiscriminantAnalysis())])
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv)
print(f"Ortalama doğruluk: {np.mean(scores):.2f}")
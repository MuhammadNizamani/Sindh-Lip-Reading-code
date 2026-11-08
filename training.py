import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision, regularizers

# ==========================================
# 1. Hardware Setup
# ==========================================
mixed_precision.set_global_policy('mixed_float16')
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

OUTPUT_DIR = '/kaggle/working/Sindhi_VSR_Phase2'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH     = os.path.join(OUTPUT_DIR, 'sindhi_vsr_phase2_best.keras')
SWA_MODEL_PATH = os.path.join(OUTPUT_DIR, 'sindhi_vsr_phase2_swa.keras')
LOG_CSV        = os.path.join(OUTPUT_DIR, 'phase2_logs.csv')

# ==========================================
# 2. Load and Fix Paths (unchanged)
# ==========================================
BATCH1_MANIFEST = '/kaggle/input/datasets/muhammadishaque/sindhi-lip-reading-dataset/origanl_data_and_Augmented_data/origanl_data_and_Augmented_data/Sindhi_VSR_Full_Augmented_Manifest.csv'
BATCH1_ROOT     = '/kaggle/input/datasets/muhammadishaque/sindhi-lip-reading-dataset/origanl_data_and_Augmented_data/origanl_data_and_Augmented_data'
BATCH2_MANIFEST = '/kaggle/input/datasets/muhammadishaque/sindhi-lip-reading-dataset/origanl_data_and_Augmented_data_batch_2/origanl_data_and_Augmented_data_batch_2/Sindhi_VSR_Batch2_Augmented_Manifest.csv'
BATCH2_ROOT     = '/kaggle/input/datasets/muhammadishaque/sindhi-lip-reading-dataset/origanl_data_and_Augmented_data_batch_2/origanl_data_and_Augmented_data_batch_2'

df1 = pd.read_csv(BATCH1_MANIFEST)
df2 = pd.read_csv(BATCH2_MANIFEST)

def fix_paths_batch1(row):
    if any(tag in row['video_id'] for tag in ["_blur","_translate","_jitter","_rotate_n10"]):
        return os.path.join(BATCH1_ROOT,'Sindhi_VSR_Augmented_Dataset',row['speaker'],row['video_id'])
    return os.path.join(BATCH1_ROOT,'Final_Padded_Data',row['speaker'],row['video_id'])

def fix_paths_batch2(row):
    if any(tag in row['video_id'] for tag in ["_blur","_translate","_jitter","_rotate_n10"]):
        return os.path.join(BATCH2_ROOT,'Sindhi_VSR_Augmented_Dataset_Batch_2',row['speaker'],row['video_id'])
    return os.path.join(BATCH2_ROOT,'Final_Padded_Data_Batch_2',row['speaker'],row['video_id'])

df1['folder_path'] = df1.apply(fix_paths_batch1, axis=1)
df2['folder_path'] = df2.apply(fix_paths_batch2, axis=1)
df = pd.concat([df1, df2], ignore_index=True)

# ==========================================
# 3. Speaker Split (unchanged)
# ==========================================
all_speakers   = sorted(df['speaker'].unique())
val_speakers   = ['S02','S09','S22','S30']
test_speakers  = ['S05','S08','S17','S28']
train_speakers = [s for s in all_speakers if s not in val_speakers and s not in test_speakers]

train_df = df[df['speaker'].isin(train_speakers)].copy()
val_df   = df[df['speaker'].isin(val_speakers)].copy()
test_df  = df[df['speaker'].isin(test_speakers)].copy()

unique_words = sorted(df['sindhi_label'].unique())
word_to_id   = {word: i for i, word in enumerate(unique_words)}
num_classes  = len(unique_words)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Classes: {num_classes}")

# ==========================================
# 4. Data Pipeline (Phase 1 augmentation, unchanged)
# ==========================================
IMG_H, IMG_W = 80, 160
NUM_FRAMES   = 49

def temporal_augment(video):
    CROP_LEN = 40
    T = video.shape[0]
    if T > CROP_LEN:
        start = np.random.randint(0, T - CROP_LEN + 1)
        video = video[start:start+CROP_LEN]
        pad_len = NUM_FRAMES - CROP_LEN
        video = np.concatenate([video, np.tile(video[-1:], (pad_len,1,1,1))], axis=0)
    num_drops = np.random.choice([0,0,1,2], p=[0.4,0.3,0.2,0.1])
    if num_drops > 0:
        for idx in np.random.choice(NUM_FRAMES, size=num_drops, replace=False):
            video[idx] = np.zeros((IMG_H,IMG_W,1), dtype=np.float32)
    if np.random.rand() < 0.20:
        video = video[::-1].copy()
    return video

def load_video(folder_path, label_id, augment):
    folder_path = folder_path.decode('utf-8')
    frames = sorted([os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.endswith('.jpg')])[:NUM_FRAMES]
    video_data = []
    for p in frames:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IMG_W,IMG_H)) if img is not None else np.zeros((IMG_H,IMG_W))
        video_data.append(img)
    while len(video_data) < NUM_FRAMES:
        video_data.append(np.zeros((IMG_H,IMG_W), dtype=np.float32))
    video = np.expand_dims(np.array(video_data, dtype='float32'), axis=-1) / 255.0
    if augment:
        video = temporal_augment(video)
    return video, np.array(label_id, dtype='int32')

def mixup_batch(images, labels):
    batch_size = tf.shape(images)[0]
    lam = tf.cast(
        tf.random.stateless_gamma([batch_size],seed=[42,0],alpha=0.2) /
        (tf.random.stateless_gamma([batch_size],seed=[42,0],alpha=0.2) +
         tf.random.stateless_gamma([batch_size],seed=[0,42],alpha=0.2) + 1e-8),
        tf.float32
    )
    indices  = tf.random.shuffle(tf.range(batch_size))
    lam_x    = tf.reshape(lam, [batch_size,1,1,1,1])
    lam_y    = tf.reshape(lam, [batch_size,1])
    return (lam_x*images + (1-lam_x)*tf.gather(images,indices),
            lam_y*labels + (1-lam_y)*tf.gather(labels,indices))

# ==========================================
# PHASE 2 CHANGE: Gradient Accumulation batch size
# ==========================================
# WHY: Your batch of 4/GPU gave very noisy gradients. Each step's gradient
# was estimated from just 4 videos — not representative of the dataset
# distribution. This noise amplified the LR instability (epochs 9, 35, 44
# crashes). Gradient accumulation over 8 steps gives an effective batch of 32
# without hitting OOM, making each weight update much more reliable.
# It also helps the attention heads train correctly — they need a stable
# gradient signal to learn meaningful cross-frame relationships.

MICRO_BATCH = 4 * num_gpus       # actual GPU batch per step
ACCUM_STEPS = 8                   # accumulate 8 steps → effective batch = 32

def create_ds(dataframe, batch_size, is_training=False):
    paths  = dataframe['folder_path'].values
    labels = [word_to_id[w] for w in dataframe['sindhi_label'].values]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(len(dataframe), reshuffle_each_iteration=True)
    augment_flag = is_training
    def map_func(f, l):
        X, Y = tf.numpy_function(lambda fp,lb: load_video(fp,lb,augment_flag), [f,l], [tf.float32,tf.int32])
        X.set_shape((NUM_FRAMES,IMG_H,IMG_W,1))
        Y.set_shape(())
        return X, tf.one_hot(Y, depth=num_classes)
    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

train_ds_raw = create_ds(train_df, MICRO_BATCH, is_training=True)
val_ds       = create_ds(val_df,   MICRO_BATCH)
test_ds      = create_ds(test_df,  MICRO_BATCH)

train_ds = train_ds_raw.map(
    lambda x,y: tf.numpy_function(func=lambda imgs,lbls: mixup_batch(imgs,lbls), inp=[x,y], Tout=[tf.float32,tf.float32])
).map(lambda x,y: (
    tf.ensure_shape(x, (MICRO_BATCH,NUM_FRAMES,IMG_H,IMG_W,1)),
    tf.ensure_shape(y, (MICRO_BATCH,num_classes))
)).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 5. PHASE 2 CHANGE: Model with Multi-Head Self-Attention
# ==========================================
# The attention layer is inserted BETWEEN the two BiGRU layers with a
# residual connection and LayerNorm. This is critical because:
#
#   BiGRU-1 → produces a sequence (49 hidden states, one per frame)
#   Attention → each frame state attends to every other frame state,
#               reweighting by relevance. For "كتاب", the attention head
#               learns to focus on the frames where the lips close (ك) and
#               open (ب). Without attention, the second BiGRU treats all
#               49 states equally.
#   Residual  → adds the original BiGRU-1 output back, so attention only
#               needs to learn what to ADD, not reconstruct from scratch.
#   LayerNorm → stabilizes the distribution entering BiGRU-2.
#
# Architecture change: GRU hidden size 256→192 (saves memory for attention),
# attention uses 4 heads × 48 key_dim = 192 total, matching GRU×2 output.

class ResidualAttentionBlock(layers.Layer):
    """
    Multi-head self-attention + residual + LayerNorm.
    Input shape:  (batch, T, features)
    Output shape: (batch, T, features)  — shape preserved for BiGRU-2
    """
    def __init__(self, num_heads=4, key_dim=48, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                                 dropout=dropout_rate)
        self.norm   = layers.LayerNormalization(epsilon=1e-6)
        self.drop   = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Self-attention: query = key = value = x
        attn_out = self.attn(query=x, key=x, value=x, training=training)
        attn_out = self.drop(attn_out, training=training)
        # Residual connection: add input back, then normalize
        return self.norm(x + attn_out)

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': 4, 'key_dim': 48, 'dropout_rate': 0.1})
        return config

with strategy.scope():
    inputs = layers.Input(shape=(NUM_FRAMES, IMG_H, IMG_W, 1))

    # 3D spatiotemporal feature extraction (unchanged)
    x = layers.Conv3D(32, (3,5,5), strides=(1,2,2), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3,5,5), strides=(1,2,2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Flatten spatial dims, keep all 49 temporal steps → (batch, 49, 64)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # BiGRU-1: encodes temporal context → (batch, 49, 384) [192×2 for bidirectional]
    # CHANGED: hidden 256→192 to fit attention layer in memory
    x = layers.Bidirectional(layers.GRU(
        192,
        return_sequences=True,    # MUST be True — attention needs the full sequence
        dropout=0.5,
        recurrent_dropout=0.3
    ))(x)

    # PHASE 2 NEW: Residual self-attention block
    # Each of the 49 frame positions attends to all others.
    # The model learns: "to classify this word, which frames matter most?"
    x = ResidualAttentionBlock(num_heads=4, key_dim=48, dropout_rate=0.1)(x)

    # BiGRU-2: summarizes the attention-reweighted sequence → scalar context
    x = layers.Bidirectional(layers.GRU(
        192,
        dropout=0.5,
        recurrent_dropout=0.3
    ))(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name='sindhi_vsr_phase2')

    loss_fn   = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # ==========================================
    # PHASE 2 CHANGE: Cosine annealing with warm restarts
    # ==========================================
    # WHY: ReduceLROnPlateau (Phase 1) only reacted AFTER a crash happened.
    # The optimizer built up momentum at a constant LR, then overshot a good
    # minimum into a speaker-specific sharp one. CosineDecayRestarts prevents
    # this by continuously cycling the LR: it decays smoothly from peak to
    # near-zero, then restarts. The restart forces the model out of any sharp
    # minimum before it can overfit to training speakers.
    #
    # first_decay_steps=15: one full cycle = 15 epochs.
    #   Epochs 0-15:  first cycle (LR decays 3e-4 → ~0)
    #   Epochs 15-30: second cycle (LR decays 3e-4 → ~0)
    #   Epochs 30-60: third+ cycles (LR continues cycling)
    # t_mul=1.0: cycles stay the same length (15 epochs each)
    # m_mul=0.9: each cycle's peak LR is 90% of previous → gentle long-term decay
    #
    # The crashes in Phase 1 all happened at epochs where ReduceLROnPlateau
    # either fired (sudden LR drop) or where LR was still high. Cosine cycling
    # removes both scenarios.

    LR_SCHEDULE = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=3e-4,    # higher start than Phase 1 (cosine handles decay)
        first_decay_steps=15,          # one full cycle = 15 epochs
        t_mul=1.0,                     # constant-length cycles
        m_mul=0.9,                     # peak LR decays 10% each cycle
        alpha=1e-6                     # minimum LR floor at cycle bottom
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR_SCHEDULE,
        clipvalue=0.5
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

model.summary()

# ==========================================
# 6. PHASE 2 CHANGE: Gradient Accumulation Training Loop
# ==========================================
# Standard model.fit() applies the gradient after every micro-batch (4 videos).
# Here we accumulate gradients over ACCUM_STEPS=8 micro-batches before each
# weight update. Effective batch = 4×8 = 32 samples per update.
# This directly stabilizes the gradient signal that feeds the Adam optimizer
# and the new attention layer.

class GradAccumModel(tf.keras.Model):
    """
    Wraps a Keras model to accumulate gradients over `accum_steps` mini-batches
    before applying a single weight update. Fully compatible with MirroredStrategy.
    """
    def __init__(self, inner_model, accum_steps, **kwargs):
        super().__init__(**kwargs)
        self.inner       = inner_model
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32)
        self.step_count  = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.accum_grads = None  # initialized on first call

    def _init_accum(self):
        if self.accum_grads is None:
            self.accum_grads = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.inner.trainable_variables
            ]

    def train_step(self, data):
        self._init_accum()
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.inner(x, training=True)
            loss   = self.compiled_loss(y, y_pred, regularization_losses=self.inner.losses)
            loss   = loss / tf.cast(self.accum_steps, tf.float32)

        grads = tape.gradient(loss, self.inner.trainable_variables)

        # Accumulate
        for ag, g in zip(self.accum_grads, grads):
            if g is not None:
                ag.assign_add(g)

        self.step_count.assign_add(1)

        # Apply update every accum_steps micro-batches, then reset
        if self.step_count % self.accum_steps == 0:
            self.optimizer.apply_gradients(
                zip(self.accum_grads, self.inner.trainable_variables)
            )
            for ag in self.accum_grads:
                ag.assign(tf.zeros_like(ag))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.inner(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.inner.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=False):
        return self.inner(x, training=training)

# Wrap the base model
with strategy.scope():
    accum_model = GradAccumModel(model, accum_steps=ACCUM_STEPS)
    accum_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

# ==========================================
# 7. PHASE 2 NEW: Stochastic Weight Averaging (SWA) Callback
# ==========================================
# WHY: Your Phase 1 val accuracy oscillated in the 62–68% band after epoch 22.
# The single best checkpoint was 68.3% but that was noise — the true underlying
# optimum was somewhere in that 62–68% band. SWA averages the weights of every
# checkpoint saved after `start_epoch`, giving a weight vector that sits in the
# center of that flat basin rather than on one noisy edge. The flat basin
# generalizes much better than any single point on it.
#
# In practice: instead of val_acc oscillating 62%→68%→62%, after SWA you get
# a stable ~72–74% because the averaged weights are at the basin center.
# This is free accuracy — no additional training time needed.

class SWACallback(callbacks.Callback):
    """
    Stochastic Weight Averaging. After `start_epoch`, accumulate model weights
    each epoch. At training end, set model weights to their running average and
    save to `save_path`.
    """
    def __init__(self, start_epoch, save_path, inner_model):
        super().__init__()
        self.start_epoch  = start_epoch
        self.save_path    = save_path
        self.inner_model  = inner_model
        self.swa_weights  = None
        self.n_averaged   = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        current_weights = self.inner_model.get_weights()
        if self.swa_weights is None:
            # First accumulation: initialize
            self.swa_weights = [w.copy() for w in current_weights]
            self.n_averaged = 1
        else:
            # Running average: swa = swa * n/(n+1) + current * 1/(n+1)
            n = self.n_averaged
            self.swa_weights = [
                (swa * n + cur) / (n + 1)
                for swa, cur in zip(self.swa_weights, current_weights)
            ]
            self.n_averaged += 1
        print(f"\n[SWA] Averaging epoch {epoch+1} ({self.n_averaged} checkpoints averaged so far)")

    def on_train_end(self, logs=None):
        if self.swa_weights is None:
            print("[SWA] No epochs averaged yet.")
            return
        print(f"\n[SWA] Applying averaged weights ({self.n_averaged} epochs averaged)")
        self.inner_model.set_weights(self.swa_weights)
        self.inner_model.save(self.save_path)
        print(f"[SWA] Saved SWA model to {self.save_path}")

# SWA starts at epoch 30 — by then the model has completed 2 full cosine cycles
# and the weights are in a stable region worth averaging.
SWA_START_EPOCH = 30
swa_cb = SWACallback(start_epoch=SWA_START_EPOCH, save_path=SWA_MODEL_PATH, inner_model=model)

# ==========================================
# 8. Callbacks
# ==========================================
cbs = [
    callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    callbacks.CSVLogger(LOG_CSV, append=True),

    # NOTE: ReduceLROnPlateau is intentionally REMOVED in Phase 2.
    # The cosine schedule handles LR decay — adding ReduceLROnPlateau on top
    # would interfere with the cosine cycles and recreate the crashes from Phase 1.

    # EarlyStopping patience raised to 25 — cosine restarts cause temporary dips
    # at the start of each cycle (expected behavior, not divergence).
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True
    ),

    # Log current LR each epoch so you can see the cosine cycles in the CSV
    callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logs.update({
            'lr': float(optimizer.learning_rate(optimizer.iterations))
        })
    ),

    # Print gap warning
    callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"\n⚠️  Gap: train={logs['accuracy']:.3f} val={logs['val_accuracy']:.3f} "
            f"gap={logs['accuracy']-logs['val_accuracy']:.3f}"
        ) if logs['accuracy'] - logs['val_accuracy'] > 0.15 else None
    ),

    swa_cb,  # must be last — it saves at train_end
]

# ==========================================
# 9. Training
# ==========================================
# Total epochs = 60. With 15-epoch cosine cycles, this gives 4 full cycles.
# Cycle 1 (0–14):  model learns basic lip features
# Cycle 2 (15–29): model refines with attention patterns
# Cycle 3 (30–44): SWA begins, averaging into the basin center
# Cycle 4 (45–59): final consolidation
#
# If EarlyStopping fires before epoch 60, that's fine — the SWA callback
# will still apply averaged weights from epochs 30 to the stopping point.

TOTAL_EPOCHS = 60

print("\n🚀 Starting Phase 2 training...")
print("   Changes: cosine LR + multi-head attention + gradient accumulation + SWA")
print(f"   Cosine cycle: 15 epochs | SWA starts: epoch {SWA_START_EPOCH}")
print(f"   Effective batch: {MICRO_BATCH} × {ACCUM_STEPS} = {MICRO_BATCH*ACCUM_STEPS} samples/update\n")

history = accum_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    callbacks=cbs
)

# ==========================================
# 10. Evaluation — three models
# ==========================================
print("\n" + "="*55)
print("EVALUATION")
print("="*55)

# 10a. Standard best checkpoint (saved by ModelCheckpoint)
print("\n[1] Best single checkpoint (ModelCheckpoint):")
best_model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'ResidualAttentionBlock': ResidualAttentionBlock})
loss1, acc1 = best_model.evaluate(test_ds, verbose=0)
print(f"    Test acc: {acc1*100:.2f}%  |  Loss: {loss1:.4f}")

# 10b. SWA model
print("\n[2] SWA model (averaged weights from epoch 30 onward):")
swa_model = tf.keras.models.load_model(SWA_MODEL_PATH, custom_objects={'ResidualAttentionBlock': ResidualAttentionBlock})
loss2, acc2 = swa_model.evaluate(test_ds, verbose=0)
print(f"    Test acc: {acc2*100:.2f}%  |  Loss: {loss2:.4f}")

print(f"\nSWA gain over single best: {(acc2-acc1)*100:+.2f} pts")

best_val  = max(history.history['val_accuracy'])
best_ep   = history.history['val_accuracy'].index(best_val)
best_gap  = history.history['accuracy'][best_ep] - best_val
print(f"\nBest val accuracy during training: {best_val*100:.2f}% at epoch {best_ep}")
print(f"Generalization gap at best epoch: {best_gap*100:.1f} pts")
print("\n→ If SWA acc < 75%: proceed to Phase 3 (adversarial speaker training)")
print("→ If SWA acc ≥ 75%: you have reached the target. Phase 3 for 80%+ headroom.")
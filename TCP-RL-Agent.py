#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import argparse
import os  # Dosya işlemleri için gerekli

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
# ns3gym modülünün tam yolunu Python'a elle gösteriyoruz
ns3gym_path = '/home/aglamazlarefe/ns-allinone-3.35/ns-3.35/contrib/opengym/model'
if ns3gym_path not in sys.path:
    sys.path.append(ns3gym_path)
    
from ns3gym import ns3env
from tcp_base import TcpTimeBased, TcpEventBased

# GPU kullanımını devre dışı bırak (CPU yeterli)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Dosya oluşturma ve loglama
try:
    w_file = open('run.log', 'w')
except:
    w_file = sys.stdout

# Argümanları ayarla
parser = argparse.ArgumentParser(description='Simülasyon betiğini başlatma/kapatma')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='ns-3 simülasyon betiğini başlatma 0/1, Varsayılan: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=100,  # Eğitim için varsayılanı artırdık
                    help='Tekrar sayısı, Varsayılan: 100')
parser.add_argument('--steps',
                    type=int,
                    default=1000, # Adım sayısını artırdık
                    help='Adım sayısı, Varsayılan: 1000')
parser.add_argument('--mode',
                    type=str,
                    default='train',
                    choices=['train', 'test'],
                    help='train: Modeli eğitir ve kaydeder. test: Kayıtlı modeli yükler ve kullanır.')

args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)

# ns-3 ortamını başlatmak için ayarları yap
port = 5555
simTime = maxSteps / 10.0 # simülasyon süresi saniye cinsinden
seed = 12
simArgs = {"--duration": str(simTime),} # str dönüşümü eklendi

dashes = "-"*18
print(f"[{dashes} Mod: {args.mode.upper()} {dashes}]")
input(f"[{dashes} Başlamak için enter'a basınız {dashes}]")

# Ortamı oluştur
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed, simArgs=simArgs)

ob_space = env.observation_space
ac_space = env.action_space

# Ajanı al veya oluştur
def get_agent(state):
    socketUuid = state[0]
    tcpEnvType = state[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            tcpAgent = TcpEventBased()
        else:
            tcpAgent = TcpTimeBased()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent
    return tcpAgent

get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

# Sinir Ağı Modeli
def modeler(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_size,)))
    model.add(tf.keras.layers.Dense((input_size + output_size) // 2, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
    return model

# Durum boyutunu belirle
state_size = ob_space.shape[0] - 4 

# --- AKSİYON AYARLARI (ROKET EKLENDİ) ---
action_size = 4  # Boyutu 4'e çıkardık
action_mapping = {}
action_mapping[0] = 0         # Sabit
action_mapping[1] = 1500      # Normal Artış
action_mapping[2] = -150      # Azalt
action_mapping[3] = 4000      # ROKET START (Hızlı Başlangıç)

# MODEL DOSYA ADI
MODEL_FILE = "tcp_rl_model.h5"

# --- MODEL YÜKLEME / OLUŞTURMA MANTIĞI ---
if args.mode == 'test':
    # TEST MODU: Eğitilmiş beyni yükle
    if os.path.exists(MODEL_FILE):
        print(f"Eğitilmiş model yükleniyor: {MODEL_FILE}")
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model yüklendi! Hazır bilgiyle başlanıyor.")
        
        # Test modunda keşif (rastgelelik) yasak!
        epsilon = 0.0 
        min_epsilon = 0.0
    else:
        print("UYARI: Model dosyası bulunamadı! Lütfen önce --mode train ile eğitim yapın.")
        sys.exit() # Programdan çık
else:
    # EĞİTİM MODU: Sıfırdan model oluştur
    print("Eğitim modu başlatılıyor. Yeni model oluşturuluyor...")
    model = modeler(state_size, action_size)
    
    # Epsilon ayarları normal (keşif açık)
    epsilon = 1.0 

# Modeli derle (Learning Rate 1e-3 yapıldı)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if args.mode == 'train':
    model.summary()

# Epsilon Decay Ayarları
epsilon_decay_param = iterationNum * 10 # Daha uzun süre keşif yapsın
min_epsilon = 0.01 # Test modunda zaten 0 olacak
epsilon_decay = (((epsilon_decay_param*maxSteps) - 1.0) / (epsilon_decay_param*maxSteps))

# Q-learning indirim faktörü
discount_factor = 0.95

# Loglama değişkenleri
total_reward = 0
reward_history = []
cWnd_history = []
rtt_history = []
tp_history = []
recency = maxSteps // 15

# --- ANA DÖNGÜ ---
for iteration in range(iterationNum):
    state = env.reset()
    state = state[4:]
    cWnd = state[1]
    init_cWnd = cWnd
    state = np.reshape(state, [1, state_size])
    
    pretty_slash = ['\\', '|', '/', '-']
    
    try:
        for step in range(maxSteps):
            # Görsellik
            pretty_index = step % 4
            print("\r[{}] Mod: {} | İterasyon: {}/{} | Dosya: {} {}".format(
                pretty_slash[pretty_index],
                args.mode.upper(),
                iteration + 1,
                iterationNum,
                w_file.name,
                '.'*(pretty_index+1)
            ), end='')

            # Epsilon-greedy seçim
            if step == 0 or np.random.rand(1) < epsilon:
                action_index = np.random.randint(0, action_size)
            else:
                action_index = np.argmax(model.predict(state)[0])

            # Aksiyon hesapla
            calc_cWnd = cWnd + action_mapping[action_index]

            # Congestion window'u sınırla (Heuristic)
            thresh = state[0][0] # ssThresh
            if step+1 > recency:
                if len(tp_history) > recency:
                    tp_dev = math.sqrt(np.var(tp_history[(-recency):]))
                    tp_1per = 0.01 * throughput
                    if tp_dev < tp_1per:
                         thresh = cWnd
            new_cWnd = max(init_cWnd, (min(thresh, calc_cWnd)))
            new_ssThresh = int(cWnd/2)
            actions = [new_ssThresh, new_cWnd]

            # Ortama gönder
            next_state, reward, done, _ = env.step(actions)
            total_reward += reward

            next_state = next_state[4:]
            cWnd = next_state[1]
            rtt = next_state[7]
            throughput = next_state[11]
            next_state = np.reshape(next_state, [1, state_size])
            
            # --- MODEL EĞİTİMİ (SADECE TRAIN MODUNDA) ---
            if args.mode == 'train':
                target = reward
                if not done:
                    target = (reward + discount_factor * np.amax(model.predict(next_state)[0]))
                
                target_f = model.predict(state)
                target_f[0][action_index] = target
                
                # Modeli güncelle
                model.fit(state, target_f, epochs=1, verbose=0)
            # ----------------------------------------------

            state = next_state
            if done: break

            if args.mode == 'train' and epsilon > min_epsilon:
                epsilon *= epsilon_decay

            # Verileri kaydet
            if iteration == iterationNum - 1: # Sadece son iterasyonu kaydet (grafik şişmesin)
                reward_history.append(total_reward)
                rtt_history.append(rtt)
                cWnd_history.append(cWnd)
                tp_history.append(throughput)
                
    finally:
        if iteration+1 == iterationNum:
            break

# --- EĞİTİM BİTTİĞİNDE MODELİ KAYDET ---
if args.mode == 'train':
    print(f"\n\nEğitim tamamlandı. Model kaydediliyor: {MODEL_FILE}")
    model.save(MODEL_FILE)
    print("Model başarıyla kaydedildi.")

# --- GRAFİK ÇİZİMİ (Sadece son turu çizer) ---
print("\nGrafikler hazırlanıyor...")
with open('rtt_tp_history.txt', 'w') as file:
    file.write("Adım\tRTT (μs)\tThroughput (bits)\n")
    for i in range(len(rtt_history)):
        file.write(f"{i+1}\t{rtt_history[i]}\t{tp_history[i]}\n")

mpl.rcdefaults()
mpl.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
plt.tight_layout(pad=4)

ax[0, 0].plot(range(len(cWnd_history)), cWnd_history)
ax[0, 0].set_title('Congestion Window')
ax[0, 0].set_xlabel('Steps')
ax[0, 0].set_ylabel('Segments')
ax[0, 0].grid(True)

ax[0, 1].plot(range(len(tp_history)), tp_history)
ax[0, 1].set_title('Throughput')
ax[0, 1].set_xlabel('Steps')
ax[0, 1].set_ylabel('Bits')
ax[0, 1].grid(True)

ax[1, 0].plot(range(len(rtt_history)), rtt_history)
ax[1, 0].set_title('RTT')
ax[1, 0].set_xlabel('Steps')
ax[1, 0].set_ylabel('Microseconds')
ax[1, 0].grid(True)

ax[1, 1].plot(range(len(reward_history)), reward_history)
ax[1, 1].set_title('Total Reward')
ax[1, 1].set_xlabel('Steps')
ax[1, 1].set_ylabel('Value')
ax[1, 1].grid(True)

plt.savefig('improved_plots.png')
print("Grafikler 'improved_plots.png' olarak kaydedildi.")
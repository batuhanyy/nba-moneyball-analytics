import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pulp
import os
from sklearn.linear_model import LinearRegression

# 1. SAYFA KONFİGÜRASYONU
st.set_page_config(page_title="NBA Moneyball Analytics", layout="wide", page_icon="🏀")

# 2. VERİ YÜKLEME VE TEMİZLEME
@st.cache_data
def load_data():
    file_path = "nba_gercek_maasli_veri.csv"
    if not os.path.exists(file_path):
        st.error(f"Kritik Hata: '{file_path}' bulunamadı. Lütfen önce 'veri_birlestir.py' çalıştırın.")
        st.stop()
    df = pd.read_csv(file_path)
    
    # Sayısal sütunları garanti altına al ve NaN değerleri temizle
    cols_to_fix = ["WS", "Salary", "Current_Salary", "PTS", "AST", "TRB", "STL", "BLK", "PER", "3PA", "TS%", "Age"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Season'] = df['Season'].astype(str)
    return df

df = load_data()

# Global Sabitler
latest_season = df['Season'].max()
active_players = df[df['Season'] == latest_season]['Player'].unique()
latest_salaries = df.groupby('Player')['Current_Salary'].first().to_dict()

# 3. YAN PANEL (SIDEBAR)
st.sidebar.title("🏀 NBA Moneyball Analytics")
st.sidebar.markdown("---")

include_retired = st.sidebar.toggle("Emekli Oyuncuları Göster", value=True)

if include_retired:
    available_players = sorted(df["Player"].dropna().unique())
    f_df = df.copy()
else:
    available_players = sorted([p for p in df["Player"].unique() if p in active_players])
    f_df = df[df["Player"].isin(active_players)].copy()

menu = st.sidebar.radio("Analiz Modülleri:", [
    "1. 👤 Oyuncu Profili & Radar Analizi",
    "2. 🏟️ Takım Kıyaslama & Benchmarking",
    "3. 📈 Piyasa Verimlilik Sınırı (Moneyball)",
    "4. 🚑 Sakatlık & Risk Senaryo Analizi",
    "5. 💎 Fiyat/Performans Keşfi (Gems)",
    "6. 🏆 AI Kadro Optimizasyonu (CBA)",
    "7. ⏳ Dönemsel Evrim Analizi (2010-2025)"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;">
        <span style="color: #808495; font-size: 12px;">PROJE KÜNYESİ</span><br>
        <b style="color: white;">Batuhan Yeniyurt</b><br>
        <b style="color: white;">Haydarhan Kirazlı</b><br>
        <b style="color: white;">Osman Oskay</b>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODÜL 1: OYUNCU PROFİLİ (Metrikler ve Radar Geri Geldi)
# ---------------------------------------------------------
if menu == "1. 👤 Oyuncu Profili & Radar Analizi":
    st.header("👤 Oyuncu Yetenek ve Değer Analizi")
    c1, c2 = st.columns(2)
    with c1: p1 = st.selectbox("Ana Oyuncuyu Seçin", available_players, index=0)
    with c2: p2 = st.selectbox("Karşılaştırılacak Oyuncu", ["Seçilmedi"] + list(available_players))

    p1_data = df[df["Player"] == p1].sort_values("Season").iloc[-1:]
    p1_maas = latest_salaries.get(p1, 2000000)
    
    st.markdown("---")
    # METRİK KARTLARI (WS, Maaş, PER, TS%)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win Shares (WS)", round(p1_data["WS"].values[0], 2))
    m2.metric("Maaş", f"${p1_maas/1000000:.1f}M")
    
    if p2 != "Seçilmedi":
        p2_data = df[df["Player"] == p2].sort_values("Season").iloc[-1:]
        p2_maas = latest_salaries.get(p2, 2000000)
        m3.metric(f"{p2} WS", round(p2_data["WS"].values[0], 2), delta=round(p2_data["WS"].values[0] - p1_data["WS"].values[0], 2))
        m4.metric(f"{p2} Maaş", f"${p2_maas/1000000:.1f}M", delta=f"${(p2_maas - p1_maas)/1000000:.1f}M", delta_color="inverse")
    else:
        m3.metric("PER (Verimlilik)", round(p1_data["PER"].values[0], 1))
        m4.metric("Şut Verimliliği (TS%)", f"%{round(p1_data['TS%'].values[0]*100, 1)}")

    

    radar_cols = ["PTS", "AST", "TRB", "STL", "BLK", "WS"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[p1_data[m].values[0] for m in radar_cols], theta=radar_cols, fill='toself', name=p1, line_color='#00FF00'))
    if p2 != "Seçilmedi":
        fig.add_trace(go.Scatterpolar(r=[p2_data[m].values[0] for m in radar_cols], theta=radar_cols, fill='toself', name=p2, line_color='#FF4B4B'))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Yetenek Radarı Karşılaştırması")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[df["Player"] == p1].sort_values("Season", ascending=False))

# ---------------------------------------------------------
# MODÜL 3: EFFICIENCY FRONTIER (Yeşil Blok Restored)
# ---------------------------------------------------------
elif menu == "3. 📈 Piyasa Verimlilik Sınırı (Moneyball)":
    st.header("📈 Piyasa Verimlilik Sınırı")
    st.success("💰 **Ekonomik Analiz:** Regresyon çizgisinin üzerinde kalan oyuncular, birim maliyet başına en yüksek galibiyet katkısını veren verimli varlıklardır.")
    
    sel_year = st.selectbox("Sezon Seç", sorted(df["Season"].unique(), reverse=True))
    year_df = f_df[(f_df["Season"] == sel_year) & (f_df["Salary"] > 0) & (f_df["WS"] > 0)].copy()
    
    if len(year_df) > 5:
        X = year_df[["Salary"]].values
        y = year_df["WS"].values
        reg = LinearRegression().fit(X, y)
        year_df["PiyasaTrendi"] = reg.predict(X)
        year_df["Verimlilik"] = year_df["WS"] - year_df["PiyasaTrendi"]
        
        fig = px.scatter(year_df, x="Salary", y="WS", hover_name="Player", color="Verimlilik",
                         color_continuous_scale="RdYlGn", size="PTS", title=f"{sel_year} Sezonu ROI Haritası")
        fig.add_traces(go.Scatter(x=year_df["Salary"], y=year_df["PiyasaTrendi"], mode='lines', name='Piyasa Ortalaması', line=dict(color='white')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Verimlilik analizi için bu sezonda yeterli veri seti bulunamadı.")

# ---------------------------------------------------------
# MODÜL 5: GEMS (KeyError 'Age' Hatası Tamamen Giderildi)
# ---------------------------------------------------------
elif menu == "5. 💎 Fiyat/Performans Keşfi (Gems)":
    st.header("💎 Moneyball Kelepirleri")
    st.success("💡 **Analiz Notu:** Düşük maaş alıp takımlarına en yüksek 'Galibiyet Payı' (WS) katkısını veren oyuncular.")
    
    max_sal = st.slider("Maksimum Maaş Sınırı (Milyon $)", 1, 15, 5) * 1000000
    gems = f_df[(f_df["Season"] == latest_season) & (f_df["Current_Salary"] <= max_sal) & (f_df["Current_Salary"] > 0)].copy()
    
    if not gems.empty:
        gems["ROI (WS/M$)"] = (gems["WS"] / (gems["Current_Salary"] / 1000000))
        gems = gems.sort_values("ROI (WS/M$)", ascending=False).head(15)
        
        fig = px.bar(gems, x="Player", y="ROI (WS/M$)", color="WS", color_continuous_scale="Greens", title="Milyon Dolar Başına WS Katkısı")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age sütununu güvenli şekilde seç
        cols = ["Player", "Current_Salary", "WS", "ROI (WS/M$)"]
        if "Age" in gems.columns:
            cols.append("Age")
        st.table(gems[cols])
    else:
        st.warning("Bu bütçe aralığında oyuncu bulunamadı.")

# ---------------------------------------------------------
# MODÜL 6: AI OPTİMİZASYONU (Yeşil Blok ve Pastası Geri Geldi)
# ---------------------------------------------------------
elif menu == "6. 🏆 AI Kadro Optimizasyonu (CBA)":
    st.header("🏆 Yapay Zeka Kadro Mühendisi (CBA)")
    st.success("""
    🍀 **CBA & Moneyball Optimizasyon Kuralları Uygulanıyor:**
    * **Maaş Tavanı (Salary Cap):** Belirlediğiniz bütçenin dışına çıkılamaz.
    * **Maksimum Kontrat (%35):** Hiçbir oyuncu tek başına bütçenin %35'inden fazlasını alamaz.
    * **Kadro Zorunluluğu:** Sistem tam 13 kişilik profesyonel bir rotasyon kurar.
    * **Harcama Alt Sınırı (%90):** NBA kuralları gereği bütçenin %90'ı harcanmalıdır.
    """)
    
    cap_m = st.slider("Maaş Tavanı (Milyon $)", 100, 250, 140)
    cap = cap_m * 1000000
    
    if st.button("🚀 Optimal Kadroyu Oluştur"):
        opt_df = f_df.groupby('Player').agg({'WS': 'mean', 'Current_Salary': 'first', 'PER': 'mean'}).reset_index()
        opt_df['Final_Salary'] = opt_df['Player'].map(latest_salaries)
        opt_df = opt_df[(opt_df['Final_Salary'] > 0) & (opt_df['Final_Salary'] <= cap * 0.35)].dropna(subset=['WS'])
        
        prob = pulp.LpProblem("NBA", pulp.LpMaximize)
        p_vars = pulp.LpVariable.dicts("P", opt_df['Player'], cat='Binary')
        
        prob += pulp.lpSum([opt_df[opt_df['Player']==p]['WS'].values[0] * p_vars[p] for p in opt_df['Player']])
        prob += pulp.lpSum([opt_df[opt_df['Player']==p]['Final_Salary'].values[0] * p_vars[p] for p in opt_df['Player']]) <= cap
        prob += pulp.lpSum([opt_df[opt_df['Player']==p]['Final_Salary'].values[0] * p_vars[p] for p in opt_df['Player']]) >= cap * 0.9
        prob += pulp.lpSum([p_vars[p] for p in opt_df['Player']]) == 13
        
        prob.solve()
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            picks = [p for p in opt_df['Player'] if p_vars[p].varValue == 1.0]
            res_df = pd.DataFrame([{"Oyuncu": p, "Maaş": f"${opt_df[opt_df['Player']==p]['Final_Salary'].values[0]/1000000:.1f}M", 
                                    "WS": round(opt_df[opt_df['Player']==p]['WS'].values[0], 2)} for p in picks])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam WS", f"{res_df['WS'].sum():.1f}")
            c2.metric("Tahmini Galibiyet", f"{min(82, res_df['WS'].sum()):.1f}")
            c3.metric("Bütçe Kullanımı", f"%{round((sum([opt_df[opt_df['Player']==p]['Final_Salary'].values[0] for p in picks])/cap)*100, 1)}")
            
            # MAAŞ PASTASI (Eksik Özellik Geri Geldi)
            fig_pie = px.pie(res_df, values=[opt_df[opt_df['Player']==p]['Final_Salary'].values[0] for p in picks], 
                             names=picks, title="Seçilen Kadronun Maaş Dağılımı", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.table(res_df.sort_values("WS", ascending=False))

# ---------------------------------------------------------
# MODÜL 7: DÖNEMSEL EVRİM (Format ve Yeşil Blok Geri Geldi)
# ---------------------------------------------------------
elif menu == "7. ⏳ Dönemsel Evrim Analizi (2010-2025)":
    st.header("⏳ NBA Oyun Karakteristiğinin Evrimi")
    st.success("📈 2010'dan günümüze NBA'de üçlük denemeleri (3PA) ve oyuncu maaşları arasındaki korelasyonun oyunun hızını (PACE) nasıl artırdığını gözlemleyin.")
    
    metric = st.selectbox("İncelemek İstediğiniz Değişim", ["3PA", "PTS", "Salary", "WS", "AST"])
    evol = df.groupby("Season")[metric].mean().reset_index().sort_values("Season")
    
    fig = px.area(evol, x="Season", y=metric, title=f"Yıllara Göre Ortalama {metric} Gelişimi", markers=True, color_discrete_sequence=['#2ecc71'])
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)
    
    val_2010 = evol.iloc[0][metric]
    val_2025 = evol.iloc[-1][metric]
    pct = ((val_2025 - val_2010) / val_2010) * 100 if val_2010 != 0 else 0
    
    # OKUNABİLİR FORMAT ($X.XX M)
    display_val = f"${val_2025/1000000:.2f} M" if metric == "Salary" else f"{val_2025:.2f}"
    st.metric(label=f"Ortalama {metric} Değişimi (2010 → 2025)", value=display_val, delta=f"%{pct:.1f}")

# DİĞER MODÜLLER (Restorasyon)
elif menu == "2. 🏟️ Takım Kıyaslama & Benchmarking":
    st.header("🏟️ Takım Tarihsel Kıyaslama")
    teams = sorted(df["Tm"].dropna().unique())
    t1 = st.selectbox("1. Takım", teams, index=0)
    t2 = st.selectbox("2. Takım", teams, index=1)
    m = st.selectbox("Metrik", ["WS", "PTS", "Salary"])
    t_data = df[df["Tm"].isin([t1, t2])].groupby(["Season", "Tm"])[m].sum().reset_index()
    st.plotly_chart(px.line(t_data, x="Season", y=m, color="Tm", markers=True), use_container_width=True)

elif menu == "4. 🚑 Sakatlık & Risk Senaryo Analizi":
    st.header("🚑 Sakatlık ve Risk Yönetimi")
    st.write("Bu modül, kadronuzdaki kritik eksikliklerin sezon sonu tablosuna etkisini simüle eder.")
    
    inj_player = st.selectbox("Sakatlanan Yıldız Oyuncu", available_players)
    missed_g = st.slider("Kaçırılacak Maç Sayısı", 1, 82, 20)
    bench_quality = st.select_slider("Yedek (Bench) Kalitesi", options=["Düşük", "Normal", "Yüksek"], value="Normal")
    
    bench_map = {"Düşük": 0.1, "Normal": 0.3, "Yüksek": 0.6}
    p_ws = df[df["Player"] == inj_player]["WS"].mean()
    # Kayıp hesaplama: Sakatlık süresince beklenen WS'nin bench kalitesi kadarını telafi edebiliyoruz
    net_loss = (p_ws * (missed_g / 82)) * (1 - bench_map[bench_quality])
    
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    s1.metric("Kayıp Galibiyet Beklentisi", f"-{round(net_loss, 2)}")
    s2.metric("Yeni Tahmini Galibiyet", round(50 - net_loss, 1))
    s3.metric("Risk Seviyesi", "Kritik 🚨" if net_loss > 4 else "Düşük ✅")

    # Monte Carlo Görselleştirme
    sims = np.random.normal(50 - net_loss, 4, 1000)
    fig = px.histogram(sims, title="1.000 Senaryoda Olası Sezon Sonu Galibiyet Sayısı", color_discrete_sequence=['red'])
    fig.add_vline(x=42, line_dash="dash", line_color="white", annotation_text="Play-off Sınırı")
    st.plotly_chart(fig, use_container_width=True)
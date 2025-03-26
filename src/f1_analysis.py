import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Настройка стилей и шрифтов
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

# Создаем папку для сохранения визуализаций
Path('visualizations').mkdir(exist_ok=True)

def load_data():
    """Загрузка всех CSV файлов с данными Формулы 1"""
    print("Загрузка данных...")
    data_dir = Path('data')
    
    # Загружаем нужные нам файлы
    dfs = {}
    required_files = [
        'circuits.csv', 'constructors.csv', 'drivers.csv',
        'races.csv', 'results.csv', 'qualifying.csv',
        'status.csv', 'pit_stops.csv'
    ]
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"Загрузка {file}...")
            dfs[file.replace('.csv', '')] = pd.read_csv(file_path)
        else:
            print(f"Файл {file} не найден!")
    
    return dfs

def prepare_data(dfs):
    """Подготовка данных для анализа"""
    print("Подготовка данных...")
    
    # Объединяем результаты гонок с информацией о гонках и пилотах
    results = dfs['results'].merge(dfs['races'], on='raceId', suffixes=('', '_race'))
    results = results.merge(dfs['drivers'], on='driverId', suffixes=('', '_driver'))
    results = results.merge(dfs['constructors'], on='constructorId', suffixes=('', '_constructor'))
    results = results.merge(dfs['circuits'], on='circuitId', suffixes=('', '_circuit'))
    
    # Объединяем с квалификацией
    results_with_quali = results.merge(
        dfs['qualifying'][['raceId', 'driverId', 'position']], 
        on=['raceId', 'driverId'], 
        how='left', 
        suffixes=('', '_quali')
    )
    
    # Объединяем со статусами для анализа причин сходов
    results_with_status = results.merge(
        dfs['status'], 
        on='statusId',
        suffixes=('', '_status')
    )
    
    return {
        'results': results,
        'results_with_quali': results_with_quali,
        'results_with_status': results_with_status,
        'pit_stops': dfs['pit_stops']
    }

def analyze_top_drivers(results, save_path='visualizations'):
    """Анализ топ-5 гонщиков по победам и их очки"""
    print("\n1. Анализ топ-5 гонщиков по победам")
    
    # Находим гонщиков с наибольшим количеством побед
    wins_by_driver = results[results['position'] == '1'].groupby(
        ['driverId', 'driverRef', 'forename', 'surname']
    ).size().sort_values(ascending=False).reset_index(name='wins')
    
    top_50_drivers = wins_by_driver.head(5)
    print(f"Топ-5 гонщиков по количеству побед:")
    for i, (_, row) in enumerate(top_5_drivers.head().iterrows(), 1):
        print(f"{i}. {row['forename']} {row['surname']}: {row['wins']} побед")
    
    # Фильтруем данные с 1980 года
    results_since_1980 = results[results['year'] >= 1980]
    
    # Получаем данные по очкам для этих гонщиков по годам
    top_drivers_id = top_5_drivers['driverId'].tolist()
    points_by_year = results_since_1980[results_since_1980['driverId'].isin(top_drivers_id)].groupby(
        ['driverId', 'driverRef', 'year']
    )['points'].sum().reset_index()
    
    # Создаем полные имена гонщиков для отображения
    driver_names = results[results['driverId'].isin(top_drivers_id)].groupby(
        ['driverId', 'driverRef', 'forename', 'surname']
    ).size().reset_index()[['driverId', 'forename', 'surname']]
    
    driver_names['fullname'] = driver_names['forename'] + ' ' + driver_names['surname']
    
    # Объединяем с полными именами
    points_by_year = points_by_year.merge(driver_names[['driverId', 'fullname']], on='driverId')
    
    # Создаем тепловую карту для топ-5 гонщиков
    top_5_drivers_id = top_5_drivers.head(5)['driverId'].tolist()
    points_for_heatmap = points_by_year[points_by_year['driverId'].isin(top_5_drivers_id)]
    
    # Создаем тепловую карту очков по годам для топ-5 гонщиков
    plt.figure(figsize=(20, 8))
    heatmap_data = points_for_heatmap.pivot_table(
        index='fullname', 
        columns='year', 
        values='points', 
        aggfunc='sum'
    ).fillna(0)
    
    # Сортируем гонщиков по количеству побед
    sorted_drivers = top_5_drivers[top_5_drivers['driverId'].isin(top_5_drivers_id)].merge(
        driver_names[['driverId', 'fullname']], on='driverId'
    ).sort_values('wins', ascending=False)['fullname'].tolist()
    
    heatmap_data = heatmap_data.reindex(sorted_drivers)
    
    # Создаем маску для нулевых значений
    mask = heatmap_data == 0
    
    # Строим тепловую карту
    ax = sns.heatmap(
        heatmap_data, 
        cmap='RdYlGn_r', 
        annot=True, 
        fmt='.0f', 
        linewidths=0.5,
        mask=mask,
        annot_kws={"size": 10}
    )
    plt.title('Очки топ-5 гонщиков по годам с 1980 года (отсортировано по количеству побед)', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{save_path}/top_drivers_points_heatmap.png', dpi=300)
    plt.close()
    
    print(f"Тепловая карта сохранена в {save_path}/top_drivers_points_heatmap.png")

def analyze_starting_position(results_with_quali, save_path='visualizations'):
    """Анализ влияния стартовой позиции на результат"""
    print("\n2. Анализ влияния стартовой позиции на результат")
    
    # Преобразуем позиции в числа
    results_with_quali['position'] = pd.to_numeric(results_with_quali['position'], errors='coerce')
    results_with_quali['position_quali'] = pd.to_numeric(results_with_quali['position_quali'], errors='coerce')
    
    # Фильтруем данные без NaN значений
    position_data = results_with_quali.dropna(subset=['position', 'position_quali'])
    
    # 2.1 Шансы на победу с поул-позиции
    pole_to_win = position_data[position_data['position_quali'] == 1]['position'].value_counts().sort_index()
    pole_win_percentage = pole_to_win.get(1, 0) / pole_to_win.sum() * 100
    
    print(f"Шансы на победу стартуя с поул-позиции: {pole_win_percentage:.2f}%")
    
    # 2.2 Создаем тепловую карту: стартовая vs финишная позиция
    # Ограничиваем анализ первыми 15 позициями для лучшей читаемости
    position_data_filtered = position_data[
        (position_data['position_quali'] <= 15) & 
        (position_data['position'] <= 15)
    ]
    
    # Создаем матрицу перехода: стартовая -> финишная позиция
    position_matrix = pd.crosstab(
        position_data_filtered['position_quali'], 
        position_data_filtered['position'],
        normalize='index'  # нормализуем по строкам
    ) * 100  # переводим в проценты
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(position_matrix, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=0.5)
    plt.title('Вероятность финишной позиции в зависимости от стартовой (в %)', fontsize=18)
    plt.xlabel('Финишная позиция')
    plt.ylabel('Стартовая позиция')
    plt.tight_layout()
    plt.savefig(f'{save_path}/starting_vs_finishing_position.png', dpi=300)
    plt.close()
    
    print(f"Визуализации сохранены в каталоге {save_path}")

def analyze_dangerous_circuits(results_with_status, save_path='visualizations'):
    """Анализ самых опасных трасс по количеству аварий и сходов"""
    print("\n3. Анализ самых опасных трасс")
    
    # Получаем список статусов, связанных с авариями и техническими проблемами
    accident_statuses = [
        'Accident', 'Collision', 'Spun off', 'Crashed', 
        'Collision damage', 'Damage', 'Debris', 'Mechanical'
    ]
    
    # Фильтруем результаты по этим статусам
    accidents = results_with_status[results_with_status['status'].str.contains('|'.join(accident_statuses), case=False)]
    
    # Считаем количество аварий/сходов по трассам
    circuit_accidents = accidents.groupby(['circuitId', 'name', 'country', 'year']).size().reset_index(name='accidents')
    
    # Считаем общее количество участников на каждой трассе в каждом году
    total_participants = results_with_status.groupby(['circuitId', 'name', 'country', 'year']).size().reset_index(name='participants')
    
    # Объединяем данные
    circuit_safety = circuit_accidents.merge(total_participants, on=['circuitId', 'name', 'country', 'year'])
    
    # Рассчитываем процент сходов/аварий
    circuit_safety['accident_rate'] = circuit_safety['accidents'] / circuit_safety['participants'] * 100
    
    # Агрегируем данные по трассам
    circuit_safety_agg = circuit_safety.groupby(['circuitId', 'name', 'country']).agg({
        'accident_rate': 'mean',
        'accidents': 'sum',
        'participants': 'sum'
    }).reset_index()
    
    # Фильтруем трассы с минимальным количеством гонок для статистической значимости
    min_participants = 100  # Минимальное количество участников
    circuit_safety_filtered = circuit_safety_agg[circuit_safety_agg['participants'] >= min_participants]
    
    # Сортируем по проценту аварий
    most_dangerous = circuit_safety_filtered.sort_values('accident_rate', ascending=False).head(15)
    
    print("Топ-5 самых опасных трасс (по проценту сходов/аварий):")
    for i, (_, row) in enumerate(most_dangerous.head().iterrows(), 1):
        print(f"{i}. {row['name']} ({row['country']}): {row['accident_rate']:.2f}% ({row['accidents']} из {row['participants']})")
    
    # Полностью закрываем все существующие фигуры
    plt.close('all')
    
    # Создаем новую визуализацию топ-15 самых опасных трасс с явным указанием figure и axes
    fig, ax = plt.subplots(figsize=(14, 10), num='dangerous_circuits', clear=True)
    
    # Сортируем для лучшего отображения
    most_dangerous = most_dangerous.sort_values('accident_rate', ascending=False)
    
    # Создаем цветовую карту
    colors = plt.cm.viridis(np.linspace(0, 1, len(most_dangerous)))
    
    # Рисуем горизонтальные бары
    y_pos = np.arange(len(most_dangerous))
    circuit_names = most_dangerous['name'] + ' (' + most_dangerous['country'] + ')'
    ax.barh(y_pos, most_dangerous['accident_rate'], color=colors)
    
    # Устанавливаем метки для оси Y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(circuit_names)
    
    # Добавляем подписи к столбцам (с четким указанием позиций)
    for i, v in enumerate(most_dangerous['accident_rate']):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    # Настраиваем внешний вид
    ax.set_title('Топ-15 самых опасных трасс Формулы 1\n(по проценту сходов и аварий)', fontsize=18)
    ax.set_xlabel('Процент сходов/аварий')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Сохраняем с явным указанием формата и разрешения
    plt.tight_layout()
    fig.savefig(f'{save_path}/most_dangerous_circuits.png', dpi=300, bbox_inches='tight')
    plt.close('all')  # Снова закрываем все фигуры
    
    print(f"Визуализации сохранены в каталоге {save_path}")

def analyze_speed_evolution(results, save_path='visualizations'):
    """Анализ эволюции скорости и влияния изменений правил"""
    print("\n4. Анализ эволюции скорости и влияния правил")
    
    # Получаем среднюю скорость победителей гонок по годам
    # Примечание: не у всех гонок есть информация о fastestLapSpeed
    speed_data = results[
        (results['position'] == '1') & 
        (results['fastestLapSpeed'].notna()) & 
        (pd.to_numeric(results['fastestLapSpeed'], errors='coerce') > 0)
    ]
    
    # Преобразуем строки в числа
    speed_data['fastestLapSpeed'] = pd.to_numeric(speed_data['fastestLapSpeed'], errors='coerce')
    
    # Группируем по годам и трассам для корректного сравнения
    yearly_speed_by_circuit = speed_data.groupby(['year', 'circuitId', 'name'])['fastestLapSpeed'].mean().reset_index()
    
    # Получаем общую среднюю скорость по годам (усредняем по трассам)
    yearly_speed = yearly_speed_by_circuit.groupby('year')['fastestLapSpeed'].mean().reset_index()
    
    # Создаем график эволюции скорости с улучшенной читабельностью
    plt.figure(figsize=(18, 10))
    
    # Настраиваем шрифты
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Строим линию с более заметными точками и улучшенным стилем
    plt.plot(yearly_speed['year'], yearly_speed['fastestLapSpeed'], 
             color='#1E88E5', linewidth=2.5, zorder=5,
             marker='o', markersize=6, markerfacecolor='white', 
             markeredgecolor='#1E88E5', markeredgewidth=2)
    
    # Создаем цветовую схему для линий изменений правил
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Добавляем отметки важных изменений в правилах
    rule_changes = {
        1994: 'Запрет активной подвески',
        2006: 'Переход на V8',
        2009: 'Аэродинамические изменения',
        2011: 'Внедрение DRS',
        2014: 'Гибридные двигатели',
        2017: 'Увеличение прижимной силы',
        2022: 'Граунд-эффект'
    }
    
    # Находим минимальное и максимальное значения скорости для размещения меток
    min_speed = yearly_speed['fastestLapSpeed'].min()
    max_speed = yearly_speed['fastestLapSpeed'].max()
    speed_range = max_speed - min_speed
    
    # Отмечаем изменения правил на графике с улучшенным форматированием
    for i, (year, change) in enumerate(rule_changes.items()):
        if year in yearly_speed['year'].values:
            color_idx = i % len(colors)
            speed_value = yearly_speed[yearly_speed['year'] == year]['fastestLapSpeed'].values[0]
            
            # Линия изменения правил - более тонкая и заметная
            plt.axvline(x=year, color=colors[color_idx], linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
            
            # Рассчитываем позицию для текста (распределяем равномерно)
            text_y_positions = np.linspace(min_speed + speed_range * 0.05, max_speed - speed_range * 0.05, len(rule_changes))
            
            # Добавляем текст с фоновым блоком для лучшей читаемости
            plt.text(year, text_y_positions[i], ' ' + change + ' ', 
                    rotation=90, ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors[color_idx], 
                            boxstyle='round,pad=0.5', linewidth=1.5))
            
            # Добавляем горизонтальную метку года для более четкой индикации
            plt.scatter([year], [speed_value], s=80, marker='o', color=colors[color_idx], 
                       zorder=10, edgecolor='white', linewidth=1.5)
    
    # Улучшаем сетку
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    
    # Форматируем оси
    plt.title('Эволюция средней скорости в Формуле 1 (2004-2024)', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Год', fontsize=16, labelpad=10)
    plt.ylabel('Средняя скорость (км/ч)', fontsize=16, labelpad=10)
    
    # Подписываем оси с более четким форматированием
    plt.xticks(np.arange(min(yearly_speed['year']), max(yearly_speed['year'])+1, 5), fontsize=12)
    
    # Добавляем аннотацию с трендом
    earliest_speed = yearly_speed.iloc[0]['fastestLapSpeed']
    latest_speed = yearly_speed.iloc[-1]['fastestLapSpeed']
    speed_change = latest_speed - earliest_speed
    percent_change = (speed_change / earliest_speed) * 100
    start_year = yearly_speed.iloc[0]['year']
    end_year = yearly_speed.iloc[-1]['year']
    
    annotation_text = f"Увеличение скорости с {start_year} по {end_year}:\n{speed_change:.1f} км/ч (+{percent_change:.1f}%)"
    plt.annotate(annotation_text, 
                xy=(0.75, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc='#F8F9FA', ec='gray', alpha=0.9),
                fontsize=12, ha='center')
    
    # Добавляем больше пространства вокруг графика
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{save_path}/speed_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ влияния DRS (введено в 2011) на количество обгонов
    # Предположим, что обгоны можно примерно оценить по изменениям позиций
    # Сравним средние изменения позиций до и после введения DRS
    
    # Преобразуем grid и position в числовые значения
    position_changes = results.copy()
    position_changes['grid'] = pd.to_numeric(position_changes['grid'], errors='coerce')
    position_changes['position'] = pd.to_numeric(position_changes['position'], errors='coerce')
    
    # Отфильтруем некорректные данные и ограничим данные с 1970 года
    position_changes = position_changes[
        (position_changes['grid'].notna()) & 
        (position_changes['position'].notna()) & 
        (position_changes['grid'] > 0) & 
        (position_changes['position'] > 0) &
        (position_changes['year'] >= 1970)  # Добавляем фильтр по году
    ]
    
    # Рассчитаем изменение позиции
    position_changes['pos_change'] = position_changes['grid'] - position_changes['position']
    
    # Группируем по годам и считаем средние изменения позиций (только положительные изменения = обгоны)
    overtakes = position_changes[position_changes['pos_change'] > 0].groupby('year')['pos_change'].agg(
        ['mean', 'sum', 'count']
    ).reset_index()
    
    # Нормализуем сумму обгонов на количество гонок в каждом году
    races_per_year = position_changes.groupby('year')['raceId'].nunique().reset_index()
    overtakes = overtakes.merge(races_per_year, on='year')
    overtakes['overtakes_per_race'] = overtakes['sum'] / overtakes['raceId']
    
    # Создаем график обгонов до и после DRS
    plt.figure(figsize=(18, 10))
    
    # Выделяем периоды до и после введения DRS
    pre_drs = overtakes[overtakes['year'] < 2011]
    post_drs = overtakes[overtakes['year'] >= 2011]
    
    plt.bar(pre_drs['year'], pre_drs['overtakes_per_race'], color='blue', alpha=0.7, label='До DRS')
    plt.bar(post_drs['year'], post_drs['overtakes_per_race'], color='green', alpha=0.7, label='После DRS')
    
    # Добавляем линию тренда
    plt.plot(overtakes['year'], overtakes['overtakes_per_race'], 'k--', alpha=0.5)
    
    # Расширенный список ключевых изменений правил (удалена строка про систему очков 2003 года)
    rule_changes = {
        1983: 'Запрет граунд-эффекта',
        1989: 'Запрет турбодвигателей',
        1994: 'Ограничения аэродинамики',
        1998: 'Узкие болиды и канавочные шины',
        2006: 'Переход на V8 двигатели',
        2009: 'Аэродинамические изменения, KERS',
        2010: 'Запрет дозаправок',
        2011: 'Внедрение DRS, шины Pirelli',
        2014: 'Гибридные двигатели V6',
        2017: 'Более широкие шины и болиды',
        2019: 'Упрощение переднего крыла',
        2021: 'Усечение днища',
        2022: 'Новый регламент (граунд-эффект)'
    }
    
    # Находим максимальное значение для размещения текста
    max_overtakes = overtakes['overtakes_per_race'].max()
    
    # Создаем цветовую схему для линий изменений
    colors = plt.cm.rainbow(np.linspace(0, 1, len(rule_changes)))
    
    # Отмечаем все изменения правил на графике
    for i, (year, change) in enumerate(rule_changes.items()):
        if year in overtakes['year'].values:
            # Используем разные цвета для лучшей различимости
            plt.axvline(x=year, color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
            
            # Вычисляем позицию для текста - распределяем равномерно
            text_height = max_overtakes * (0.4 + 0.5 * i / len(rule_changes))
            
            # Ограничиваем высоту текста
            text_height = min(text_height, max_overtakes * 0.95)
            
            plt.text(year, text_height, change, rotation=90, ha='center', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[i], boxstyle='round,pad=0.3'))
    
    plt.title('Влияние изменений правил на количество обгонов (1970-2024)', fontsize=18)
    plt.xlabel('Год', fontsize=12)
    plt.ylabel('Среднее количество обгонов на гонку', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}/overtakes_evolution.png', dpi=300)
    plt.close()
    
    print(f"Визуализации сохранены в каталоге {save_path}")


def main():
    # Загружаем данные
    dfs = load_data()
    
    # Проверяем, что все нужные данные загружены
    required_dfs = ['circuits', 'constructors', 'drivers', 'races', 'results', 'qualifying', 'status', 'pit_stops']
    missing_dfs = [df for df in required_dfs if df not in dfs]
    
    if missing_dfs:
        print(f"Ошибка: следующие необходимые файлы данных отсутствуют: {', '.join(missing_dfs)}")
        return
    
    # Подготавливаем данные
    prepared_data = prepare_data(dfs)
    
  
    
    print("\nАнализ завершен! Все визуализации сохранены в каталоге 'visualizations'")

if __name__ == "__main__":
    main() 

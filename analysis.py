def answer_wikipedia_questions(df, plot_function):
    # Q1: $2B movies before 2000 (corrected from 2020)
    q1 = df[(df['worldwide_gross'] >= 2_000_000_000) & (df['year'] < 2000)].shape[0]

    # Q2: Earliest film > $1.5B with fallback column
    sort_col = 'year' if 'year' in df.columns else 'release_date'
    q2_df = df[df['worldwide_gross'] > 1_500_000_000].sort_values(by=sort_col)
    q2 = q2_df.iloc[0]['title'] if not q2_df.empty else "N/A"

    # Q3: Correlation between Rank and Peak
    rank_peak_df = df[['rank', 'peak']].dropna()
    q3 = rank_peak_df['rank'].corr(rank_peak_df['peak'])

    # Q4: Scatterplot
    q4 = plot_function(df, x_col='rank', y_col='peak', title='Rank vs Peak')

    return [q1, q2, round(q3, 6), q4]

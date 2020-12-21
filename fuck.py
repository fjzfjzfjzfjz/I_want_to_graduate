import pickle

m_tw = pickle.load(open('preprocessed_data/twitter/mapping.pickle', 'rb'))
m_fq = pickle.load(open('preprocessed_data/foursquare/mapping.pickle', 'rb'))
m_fb = pickle.load(open('preprocessed_data/facebook/mapping.pickle', 'rb'))
aligned_fb_fq = pickle.load(open('preprocessed_data/aligned_fb_fq.pickle', 'rb'))
aligned_fq_tw = pickle.load(open('preprocessed_data/aligned_fq_tw.pickle', 'rb'))
aligned_fb_tw = pickle.load(open('preprocessed_data/aligned_fb_tw.pickle', 'rb'))


def c(a, m1, m2):
    rt = []
    for i in a:
        k1 = int(i[0])
        k2 = int(i[1])
        if k1 in m1 and k2 in m2:
            rt.append([m1[k1], m2[k2]])
    return rt


aligned_fb_tw1 = c(aligned_fb_tw, m_fb, m_tw)
aligned_fb_fq1 = c(aligned_fb_fq, m_fb, m_fq)
aligned_fq_tw1 = c(aligned_fq_tw, m_fq, m_tw)
pickle.dump(aligned_fb_fq1, open('preprocessed_data/aligned_fb_fq_final.pickle', 'wb'))
pickle.dump(aligned_fb_tw1, open('preprocessed_data/aligned_fb_tw_final.pickle', 'wb'))
pickle.dump(aligned_fq_tw1, open('preprocessed_data/aligned_fq_tw_final.pickle', 'wb'))

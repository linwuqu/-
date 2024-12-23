app是主框架 gr主界面的代码基本都在这里 模型位置从`https://huggingface.co/spaces/danielhajialigol/DRGCoder/tree/main`这里的app.py进行下载（联网下好后 记得改成自己的本地文件 不然总会报错） 其他LFS文件也都是在那里跑不起来可能是文件没下全
embeddings_cpu.pt和cleaned_patients.csv是重新做的 待会我看看能不能上传上来

目前是文档主题检测也做完了 图也都可以画了 但是具体拼起来的过程还在fys那里

`https://huggingface.co/spaces/theArijitDas/Product-Description-Similarity/tree/main` 相似度参考
`https://huggingface.co/spaces/brendenc/Hip-Hop-gRadio` umap可视化参考
主题提取（分类）用的NMF `n_topic`=10
top-k用的pyLDAvis 直接对lda结果可视化展示

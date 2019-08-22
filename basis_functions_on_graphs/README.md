<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>


## Two Preliminary Demos

Euclidean space is huge, and the dataset of interest often occupies a very small manifold within Euclidean space. Often, distance in terms of that manifold is more significant than distance in Euclidean space.

Take, as a concrete example, the swiss roll dataset.

Consider a point $x_1$ on the outside tip of the 'roll'. Now consider a point $x_2$ directly below $x_1$. Finally, consider a point $x_3$ farther below $x_1$ at the bottom of the swiss roll. In Euclidean distance, $x_1$ is closer to $x_2$ than to $x_3$. However, if we were to 'un-roll' the swiss roll, we would find that $x_1$ is much closer to $x_3$ than to $x_2$. This is an example where data lies on a manifold that isn't measured well by Euclidean distance.

Why is this relevant to machine learning?

Suppose we have a set of inputs that we believe are related to a set of outputs -- let's say, height $H$ and salary $S$ -- by an unknown function $f$. We sample a training set ${X, Y} \subset H \times S $ of heights and salaries. The inductive bias of machine learning is that a true function $f$ maps points close together in $X$ to points close together in $Y$. So if we can find a function $\hat{f}$ that predicts outputs $Y$ for inputs $X$, then $\hat{f}$ will map new points close to $X$ to the correct region in $S$.

But what if our distance measure is off? Take our points $x_1$ and $x_2$ from the previous example, and suppose they are from the domain $H$ of heights. They are close in Euclidean distance but far in terms of distance on the manifold. If $\hat{f}$ is a machine learning model that is trained on representations of $X$ in Euclidean space, it will map $x_1$ and $x_2$ to similar points in the codomain $S$, when they should map to totally different parts of the space. 

close to $X$ to points  in $X$ will m,  we can learn to predict outputs for kn

In short, we believe that we can generalize beyond the training data.
As a preface to conducting learning on graphs,

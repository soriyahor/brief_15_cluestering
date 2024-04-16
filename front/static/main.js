
document.getElementById('btn-submit').addEventListener('click', (event) => {
    event.preventDefault();
    const selectedModel = document.getElementById('modelDropdown').value;
    const selectedClusters = document.getElementById('n_clusters').value;


    fetch('http://soriyab15-fastfront.francecentral.azurecontainer.io:8000/choose_model/' + selectedModel + '?n_clusters=' + selectedClusters, {
        method: 'POST'
    })

    .then(response => response.text()) 
    .then(text => { 
        let resObject = JSON.parse(text);
        let res = JSON.parse(resObject);
        document.getElementById('result').innerHTML = "<h2>Score " +selectedModel +"</h2><p>Sihouette : " + res.silhouette_score + "</p><p>David Bouldin : " + res.davies_bouldin_score + "</p>";
        showImage(res.plot_image);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function showImage(base64Image) {
    // Supprimer l'image précédente
    const resultPlot = document.getElementById('result_plot');
    while (resultPlot.firstChild) {
        resultPlot.removeChild(resultPlot.firstChild);
    }

    let img = document.createElement('img');
    img.src = "data:image/png;base64," + base64Image;
    document.getElementById('result_plot').appendChild(img);
}


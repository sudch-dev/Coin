
function fetchData() {
    fetch("/api/fetch-data", {method: "POST"})
        .then(() => fetch("/api/trade-status"))
        .then(res => res.json())
        .then(data => {
            document.getElementById("output").textContent = JSON.stringify(data, null, 2);
        });
}
fetchData();

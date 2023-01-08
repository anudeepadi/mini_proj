const stext =document.querySelector('.container')


fetch("http://localhost:8000/getFromMongo",{
    method: 'GET',
    headers: {
        "Content-type": "application/json"
    }
})
.then(res=>res.json())
.then(data=>{
    console.log(data)
    console.log(data.collections)
    let serverText=data.collections
    console.log(typeof(serverText))
    for(let i =0; i<serverText.length;i++){
        stext.innerHTML+=`
       
        <div class="text">
        <p>summary :</p>
        <p>${serverText[i].summary}</p>
        <p>original text :</p>
        <p>${serverText[i].text}</>
        </div>`
    }
})
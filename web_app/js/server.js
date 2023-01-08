const express = require("express")
const app = express()
const cors =require("cors")
const { json } = require("body-parser")
app.use(json)
app.use(cors)

app.post('/summarize', async (req,res)=>{

    const data= req.body
    await fetch(`http://127.0.0.1:8000/summarize?text=?${stext.value}`, {
    method: 'POST',
    headers: {
        "Content-type": "application/json"
    }
}).then(res=>res.json())
.then(data=>{
    console.log(data)

    const values = Object.values(data);
   
    console.log(values)
    const res =JSON.stringify(data)
    console.log(res)
    console.log(data)
    console.log(Object.keys(data));
    resText.innerHTML=`<p class ="lead">
    <b> Original Text </b> : ${values[0]}
    <br>
    <br>
    <br>
   <strong> Summary text</strong>  : ${values[1]}
    <br>
   <b> Length Before Summariztion</b> : ${values[2]}
    <br>
    <b> Length After Summarization </b> : ${values[3]}
     <br>
    <b> Percentage Reduction</b> : ${values[4]}
     <br>
   <b> time taken </b> : ${values[5]}
                          </p>`
                         
})
})

app.listen(5050,()=>{
    console.log("server is up at 5050")
})
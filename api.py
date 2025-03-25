from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import base64
import tempfile
import time
from pathlib import Path
from interface import ChatInterface
from main import initialize_agent
import os

app = FastAPI(title="MedRAX API", 
              description="REST API for MedRAX medical imaging analysis")

# Initialize agent (same as main.py)
selected_tools = [
    "ImageVisualizerTool",
    "DicomProcessorTool", 
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    "ChestXRayReportGeneratorTool",
    "XRayVQATool"
    # "LlavaMedTool",
    # "XRayPhraseGroundingTool",
    # "ChestXRayGeneratorTool",
]

openai_kwargs = {}
if api_key := os.getenv("OPENAI_API_KEY"):
    openai_kwargs["api_key"] = api_key
if base_url := os.getenv("OPENAI_BASE_URL"):
    openai_kwargs["base_url"] = base_url

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="gpt-4o-mini",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs=openai_kwargs
)

interface = ChatInterface(agent, tools_dict)

@app.post("/inference")
async def single_inference(file: UploadFile = File(...)):
    """Process a single medical image"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        # Process through interface
        display_path = interface.handle_upload(temp_path)
        
        # Get inference results
        messages = [{"role": "user", "content": f"image_path: {temp_path}"}]
        response = agent.workflow.invoke(
            {"messages": messages},
            {"configurable": {"thread_id": str(time.time())}}
        )
        
        # Clean up
        Path(temp_path).unlink()
        
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            
            def process_message(msg):
                if hasattr(msg, 'content'):
                    return {
                        "content": msg.content,
                        "additional_kwargs": getattr(msg, 'additional_kwargs', {}),
                        "type": type(msg).__name__
                    }
                elif isinstance(msg, (dict, list, str, int, float, bool)):
                    return msg
                else:
                    return str(msg)

            if isinstance(response, dict):
                processed = {k: process_message(v) for k, v in response.items()}
            elif hasattr(response, '__dict__'):
                processed = process_message(response)
            else:
                processed = str(response)

            logger.info(f"Processed response: {processed}")
            return JSONResponse({
                "status": "success",
                "result": processed,
                "display_image": display_path
            })
            
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_inference") 
async def batch_inference(files: List[UploadFile] = File(...)):
    """Process multiple medical images"""
    results = []
    for file in files:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Process through interface
            display_path = interface.handle_upload(temp_path)
            
            # Get inference results
            messages = [{"role": "user", "content": f"image_path: {temp_path}"}]
            response = agent.workflow.invoke(
                {"messages": messages},
                {"configurable": {"thread_id": str(time.time())}}
            )
            
            try:
                logger.info(f"Batch response type: {type(response)}")
                logger.info(f"Batch response attributes: {dir(response)}")
                
                if isinstance(response, dict):
                    processed = {k: process_message(v) for k, v in response.items()}
                elif hasattr(response, '__dict__'):
                    processed = process_message(response)
                else:
                    processed = str(response)

                logger.info(f"Processed batch response: {processed}")
                results.append({
                    "filename": file.filename,
                    "result": processed,
                    "display_image": display_path
                })
                
            except Exception as e:
                logger.error(f"Error processing batch response: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": f"Response processing error: {str(e)}"
                })
            
            # Clean up
            Path(temp_path).unlink()
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "status": "completed",
        "results": results
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8585)

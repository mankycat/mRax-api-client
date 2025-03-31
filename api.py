from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import base64
import tempfile
import time
from pathlib import Path
from interface import ChatInterface
from main import initialize_agent
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_message(msg):
    """Recursively process message objects into serializable format"""
    try:
        if hasattr(msg, 'content'):
            result = {
                "content": msg.content,
                "type": type(msg).__name__
            }
            if hasattr(msg, 'additional_kwargs'):
                result["additional_kwargs"] = {
                    k: v for k, v in msg.additional_kwargs.items()
                    if isinstance(v, (str, int, float, bool, dict, list)) or v is None
                }
            return result
        elif isinstance(msg, (dict, list)):
            return {k: process_message(v) for k, v in msg.items()} if isinstance(msg, dict) \
                   else [process_message(v) for v in msg]
        elif isinstance(msg, (str, int, float, bool)) or msg is None:
            return msg
        elif hasattr(msg, '__dict__'):
            return {k: process_message(v) for k, v in msg.__dict__.items()}
        else:
            return str(msg)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {"error": str(e), "original_type": type(msg).__name__}

app = FastAPI(title="MedRAX API", 
              description="REST API for MedRAX medical imaging analysis")

# Initialize agent (same as main.py)
selected_tools = [
    "ImageVisualizerTool",
    # "DicomProcessorTool", 
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    "ChestXRayReportGeneratorTool",
    "XRayVQATool",
    "LlavaMedTool",
    "XRayPhraseGroundingTool",
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

# Print available tools for debugging
logger.info("Available tools:")
for tool_name, tool in tools_dict.items():
    logger.info(f"Tool: {tool_name}, API Name: {getattr(tool, 'name', 'unknown')}")

interface = ChatInterface(agent, tools_dict)

@app.post("/inference")
async def single_inference(
    file: UploadFile = File(...), 
    user_message: str = Form(None), 
    force_tool: str = Form(None)
):
    """Process a single medical image with optional user message"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        # Process through interface
        display_path = interface.handle_upload(temp_path)
        
        # Log received parameters for debugging
        logger.info(f"Received parameters: file={file.filename}, user_message={user_message}, force_tool={force_tool}")
        
        # Get inference results
        messages = []
        
        # Send path for tools (like in the original implementation)
        messages.append({"role": "user", "content": f"image_path: {temp_path}"})
        
        # Load and encode image for multimodal processing
        with open(temp_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        
        # Add the image as a multimodal message (exactly as in the original implementation)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        )
        
        # Format user message exactly as in the original implementation
        if force_tool:
            # If force_tool is specified, add a special instruction to use that tool
            tool_instruction = f"Please analyze this image using the {force_tool} tool."
            if user_message:
                # Combine the tool instruction with the user message
                combined_message = f"{tool_instruction} {user_message}"
                messages.append({"role": "user", "content": [{"type": "text", "text": combined_message}]})
            else:
                # Just use the tool instruction
                messages.append({"role": "user", "content": [{"type": "text", "text": tool_instruction}]})
        elif user_message:
            # Normal case, just use the user message
            messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
        
        # Log the final messages array for debugging
        logger.info(f"Request messages: {messages}")

        response = None
        processed = None

        if force_tool:
            logger.info(f"Attempting to force tool: {force_tool}")
            tool_to_force = tools_dict.get(force_tool)
            if tool_to_force:
                try:
                    # Prepare input for the tool - adjust this based on actual tool needs
                    # The error log indicates LlavaMedTool expects 'question', not 'query'.
                    tool_input = {"image_path": temp_path}
                    if user_message:
                        tool_input["question"] = user_message # Changed 'query' to 'question'
                    
                    logger.info(f"Invoking tool '{force_tool}' directly with input: {tool_input}")
                    # Assuming tools have a standard 'run' method. Adjust if needed (_run, _arun, etc.)
                    # We might need to run this synchronously if the tool doesn't support async
                    # For simplicity, let's assume a synchronous run method exists
                    tool_response = tool_to_force.run(tool_input) 
                    logger.info(f"Direct tool response type: {type(tool_response)}")
                    
                    # Process the direct tool response
                    processed = process_message({"tool_output": tool_response}) # Wrap in a dict for consistency?

                except Exception as tool_error:
                    logger.error(f"Error directly invoking tool {force_tool}: {str(tool_error)}")
                    raise HTTPException(status_code=500, detail=f"Error executing forced tool: {str(tool_error)}")
                finally:
                    # Clean up temp file even if tool fails
                    Path(temp_path).unlink()
            else:
                Path(temp_path).unlink() # Clean up before raising error
                logger.error(f"Invalid force_tool specified: {force_tool}. Available tools: {list(tools_dict.keys())}")
                raise HTTPException(status_code=400, detail=f"Invalid force_tool specified: {force_tool}")
        else:
            # Original workflow using the agent
            response = agent.workflow.invoke(
                {"messages": messages},
                {"configurable": {"thread_id": str(time.time())}}
            )
            # Clean up temp file after agent invocation
            Path(temp_path).unlink()
            
            # Process agent response if not already processed by tool forcing
            if response:
                 try:
                    logger.info(f"Agent response type: {type(response)}")
                    logger.info(f"Agent response attributes: {dir(response)}")
                    
                    if isinstance(response, dict):
                        processed = {k: process_message(v) for k, v in response.items()}
                    elif hasattr(response, '__dict__'):
                        processed = process_message(response)
                    else:
                        processed = str(response)
                 except Exception as e:
                    logger.error(f"Error processing agent response: {str(e)}")
                    # Don't raise here, let the outer try-except handle it if needed
                    processed = {"error": f"Response processing error: {str(e)}"}


        # Ensure 'processed' is not None before returning
        if processed is None:
             # This case should ideally not happen if logic is correct, but as a safeguard:
             logger.error("Response processing failed, 'processed' is None.")
             raise HTTPException(status_code=500, detail="Internal server error: Failed to process response.")

        # The 'processed' variable now holds the result from either the forced tool or the agent workflow.
        logger.info(f"Final processed response: {processed}")
        return JSONResponse({
            "status": "success",
                "result": processed,
                "display_image": display_path # display_path is generated before the if/else
            })

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to return proper status codes
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in single_inference: {str(e)}", exc_info=True)
        # Ensure temp file is cleaned up in case of unexpected error before unlink
        if 'temp_path' in locals() and Path(temp_path).exists():
             Path(temp_path).unlink()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/batch_inference") 
async def batch_inference(
    files: List[UploadFile] = File(...), 
    user_message: str = Form(None), 
    force_tool: str = Form(None)
):
    """Process multiple medical images with optional user message"""
    results = []
    for file in files:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Process through interface
            display_path = interface.handle_upload(temp_path)
            
            # Log received parameters for batch processing
            logger.info(f"Batch processing: file={file.filename}, user_message={user_message}, force_tool={force_tool}")
            
            # Get inference results
            messages = []
            
            # Send path for tools (like in the original implementation)
            messages.append({"role": "user", "content": f"image_path: {temp_path}"})
            
            # Load and encode image for multimodal processing
            with open(temp_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Add the image as a multimodal message (exactly as in the original implementation)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                }
            )
            
            # Format user message exactly as in the original implementation
            if force_tool:
                # If force_tool is specified, add a special instruction to use that tool
                tool_instruction = f"Please analyze this image using the {force_tool} tool."
                if user_message:
                    # Combine the tool instruction with the user message
                    combined_message = f"{tool_instruction} {user_message}"
                    messages.append({"role": "user", "content": [{"type": "text", "text": combined_message}]})
                else:
                    # Just use the tool instruction
                    messages.append({"role": "user", "content": [{"type": "text", "text": tool_instruction}]})
            elif user_message:
                # Normal case, just use the user message
                messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
            # Log the final messages array for batch processing
            logger.info(f"Batch request messages: {messages}")

            response = None
            processed = None

            if force_tool:
                logger.info(f"Batch: Attempting to force tool: {force_tool}")
                tool_to_force = tools_dict.get(force_tool)
                if tool_to_force:
                    try:
                        # Prepare input for the tool
                        # The error log indicates LlavaMedTool expects 'question', not 'query'.
                        tool_input = {"image_path": temp_path}
                        if user_message:
                            tool_input["question"] = user_message # Changed 'query' to 'question'
                        
                        logger.info(f"Batch: Invoking tool '{force_tool}' directly with input: {tool_input}")
                        tool_response = tool_to_force.run(tool_input)
                        logger.info(f"Batch: Direct tool response type: {type(tool_response)}")
                        
                        # Process the direct tool response
                        processed = process_message({"tool_output": tool_response})

                    except Exception as tool_error:
                        logger.error(f"Batch: Error directly invoking tool {force_tool} for {file.filename}: {str(tool_error)}")
                        # Store error for this specific file
                        processed = {"error": f"Error executing forced tool: {str(tool_error)}"}
                    # No finally block for unlink here, it's handled after the if/else block
                else:
                    logger.error(f"Batch: Invalid force_tool specified: {force_tool} for {file.filename}. Available tools: {list(tools_dict.keys())}")
                    # Store error for this specific file
                    processed = {"error": f"Invalid force_tool specified: {force_tool}"}
            else:
                 # Original workflow using the agent
                response = agent.workflow.invoke(
                    {"messages": messages},
                    {"configurable": {"thread_id": str(time.time())}}
                )
                # Process agent response if not already processed by tool forcing
                if response:
                    try:
                        logger.info(f"Batch agent response type: {type(response)}")
                        logger.info(f"Batch agent response attributes: {dir(response)}")
                        # Process the response *inside* the try block
                        if isinstance(response, dict):
                            processed = {k: process_message(v) for k, v in response.items()}
                        elif hasattr(response, '__dict__'):
                            processed = process_message(response)
                        else:
                            processed = str(response)
                    except Exception as e:
                        logger.error(f"Batch: Error processing agent response for {file.filename}: {str(e)}")
                        processed = {"error": f"Response processing error: {str(e)}"}

            # Ensure 'processed' is not None before appending results
            if processed is None:
                logger.error(f"Batch: Response processing failed for {file.filename}, 'processed' is None.")
                processed = {"error": "Internal server error: Failed to process response."}


            logger.info(f"Processed batch response for {file.filename}: {processed}")
            results.append({
                "filename": file.filename,
                "result": processed,
                "display_image": display_path
            })

            # Clean up temp file regardless of success/failure within the loop iteration
            Path(temp_path).unlink()

        except Exception as e:
            logger.error(f"Error processing file {file.filename} in batch: {str(e)}", exc_info=True)
            results.append({
                "filename": file.filename,
                "error": f"Processing error: {str(e)}"
            })
            # Ensure temp file is cleaned up if error occurred before unlink
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
    
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

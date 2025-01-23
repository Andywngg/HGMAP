import pandas as pd
from docx import Document
from pathlib import Path
import logging
from typing import Union, List, Dict
import re

logger = logging.getLogger(__name__)

class DataConverter:
    """Utility class to convert various file formats to CSV"""
    
    @staticmethod
    def excel_to_csv(
        excel_path: Union[str, Path],
        output_path: Union[str, Path],
        sheet_name: Union[str, int, None] = 0
    ) -> Path:
        """
        Convert Excel file to CSV
        
        Args:
            excel_path: Path to Excel file
            output_path: Path to save CSV file
            sheet_name: Name or index of sheet to convert (default: 0, first sheet)
        
        Returns:
            Path to saved CSV file
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Clean column names
            df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
            
            # Remove any problematic characters
            df = df.replace({r'[\\/*?:"<>|]': ''}, regex=True)
            
            # Save to CSV
            output_path = Path(output_path)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Successfully converted {excel_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting Excel file: {e}")
            raise
    
    @staticmethod
    def word_to_dataframe(
        docx_path: Union[str, Path]
    ) -> List[pd.DataFrame]:
        """
        Extract tables from Word document into DataFrames
        
        Args:
            docx_path: Path to Word document
        
        Returns:
            List of pandas DataFrames, one for each table found
        """
        try:
            # Read Word document
            doc = Document(docx_path)
            
            # Extract tables
            dataframes = []
            for table in doc.tables:
                # Get headers from first row
                headers = [cell.text.strip().replace(' ', '_').lower() 
                          for cell in table.rows[0].cells]
                
                # Get data from remaining rows
                data = []
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    data.append(row_data)
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=headers)
                dataframes.append(df)
            
            logger.info(f"Successfully extracted {len(dataframes)} tables from {docx_path}")
            return dataframes
            
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            raise
    
    @staticmethod
    def word_to_csv(
        docx_path: Union[str, Path],
        output_dir: Union[str, Path],
        table_names: List[str] = None
    ) -> List[Path]:
        """
        Convert tables in Word document to CSV files
        
        Args:
            docx_path: Path to Word document
            output_dir: Directory to save CSV files
            table_names: Names to use for CSV files (default: table_1, table_2, etc.)
        
        Returns:
            List of paths to saved CSV files
        """
        try:
            # Extract tables
            dataframes = DataConverter.word_to_dataframe(docx_path)
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate table names if not provided
            if table_names is None:
                table_names = [f"table_{i+1}" for i in range(len(dataframes))]
            
            # Save each table as CSV
            csv_paths = []
            for df, name in zip(dataframes, table_names):
                output_path = output_dir / f"{name}.csv"
                df.to_csv(output_path, index=False)
                csv_paths.append(output_path)
            
            logger.info(f"Successfully saved {len(csv_paths)} CSV files to {output_dir}")
            return csv_paths
            
        except Exception as e:
            logger.error(f"Error saving CSV files: {e}")
            raise
    
    @staticmethod
    def convert_all(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, List[Path]]:
        """
        Convert all supported files in a directory to CSV
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save CSV files
        
        Returns:
            Dictionary mapping original files to their converted CSV files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        conversion_map = {}
        
        # Process Excel files
        for excel_file in input_dir.glob("*.xlsx"):
            output_path = output_dir / f"{excel_file.stem}.csv"
            converted_path = DataConverter.excel_to_csv(excel_file, output_path)
            conversion_map[str(excel_file)] = [converted_path]
        
        # Process Word files
        for word_file in input_dir.glob("*.docx"):
            word_output_dir = output_dir / word_file.stem
            csv_paths = DataConverter.word_to_csv(word_file, word_output_dir)
            conversion_map[str(word_file)] = csv_paths
        
        return conversion_map 
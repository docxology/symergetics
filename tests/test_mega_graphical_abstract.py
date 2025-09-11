#!/usr/bin/env python3
"""
Tests for mega graphical abstract functionality.

This test file focuses on real functionality testing without mocks,
ensuring the mega graphical abstract creates actual files and works correctly.
"""

import pytest
import tempfile
from pathlib import Path
from symergetics.visualization import create_mega_graphical_abstract


class TestMegaGraphicalAbstract:
    """Test mega graphical abstract functionality."""

    def test_mega_graphical_abstract_creation(self):
        """Test that mega graphical abstract creates a real file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_mega_abstract.png'
            
            result = create_mega_graphical_abstract(
                title="Test Mega Graphical Abstract",
                output_path=str(output_path),
                backend="matplotlib"
            )
            
            # Check result structure
            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'mega_graphical_abstract'
            assert result['metadata']['panels'] == 17
            assert 'Test Mega Graphical Abstract' in result['metadata']['title']
            
            # Check that file was actually created
            assert len(result['files']) == 1
            created_file = Path(result['files'][0])
            assert created_file.exists()
            assert created_file.stat().st_size > 0  # File has content
            
            # Check file extension
            assert created_file.suffix == '.png'

    def test_mega_graphical_abstract_default_path(self):
        """Test mega graphical abstract with default path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to test default path generation
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                result = create_mega_graphical_abstract(
                    title="Default Path Test",
                    backend="matplotlib"
                )
                
                # Should create file in organized structure
                assert len(result['files']) == 1
                created_file = Path(result['files'][0])
                assert created_file.exists()
                
                # Should be in mathematical directory
                assert 'mathematical' in str(created_file)
                assert 'mega_graphical_abstract' in str(created_file)
                
            finally:
                os.chdir(original_cwd)

    def test_mega_graphical_abstract_metadata(self):
        """Test that mega graphical abstract has correct metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_title = "Custom Test Title for Mega Abstract"
            
            result = create_mega_graphical_abstract(
                title=custom_title,
                output_path=Path(temp_dir) / 'custom_test.png',
                backend="matplotlib"
            )
            
            # Check metadata
            metadata = result['metadata']
            assert metadata['type'] == 'mega_graphical_abstract'
            assert metadata['title'] == custom_title
            assert metadata['panels'] == 17
            assert 'dimensions' in metadata
            assert 'description' in metadata
            assert '32x24 inches' in metadata['dimensions']
            assert 'Comprehensive visual overview' in metadata['description']

    def test_mega_graphical_abstract_invalid_backend(self):
        """Test mega graphical abstract with invalid backend raises error."""
        with pytest.raises(ValueError, match="Mega graphical abstract requires matplotlib backend"):
            create_mega_graphical_abstract(
                backend="invalid_backend"
            )

    def test_mega_graphical_abstract_file_content(self):
        """Test that the created file has reasonable size (not empty or corrupted)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'content_test.png'
            
            result = create_mega_graphical_abstract(
                title="Content Test",
                output_path=str(output_path),
                backend="matplotlib"
            )
            
            created_file = Path(result['files'][0])
            
            # PNG files should be at least a few KB for a complex plot
            file_size_kb = created_file.stat().st_size / 1024
            assert file_size_kb > 50  # Should be at least 50KB for a detailed plot
            
            # Should be a valid PNG file (check magic bytes)
            with open(created_file, 'rb') as f:
                header = f.read(8)
                assert header.startswith(b'\x89PNG')  # PNG magic bytes

    @pytest.mark.parametrize("title", [
        "Simple Title",
        "Complex Title with Special Characters: @#$%^&*()",
        "Very Long Title That Should Still Work Properly And Not Cause Any Issues With Layout",
        "Title with\nNewlines",
        "Title with Unicode: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω"
    ])
    def test_mega_graphical_abstract_various_titles(self, title):
        """Test mega graphical abstract with various title formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_mega_graphical_abstract(
                title=title,
                output_path=Path(temp_dir) / 'title_test.png',
                backend="matplotlib"
            )
            
            # Should succeed regardless of title content
            assert len(result['files']) == 1
            created_file = Path(result['files'][0])
            assert created_file.exists()
            assert result['metadata']['title'] == title
